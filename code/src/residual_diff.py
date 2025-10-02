import torch
from torch import nn
import torch.nn.functional as F
from collections import namedtuple
from tqdm.auto import tqdm
from functools import partial
from einops import reduce

from .utils import *
from .textModel import LanguageModel

ModelResPrediction = namedtuple('ModelResPrediction', ['pred_res', 'pred_noise', 'pred_x_start'])

class ResidualDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps=1000,
        sampling_timesteps=None,
        loss_type='l1',
        objective='pred_res_noise',
        ddim_sampling_eta=0.,
        sum_scale=None,
        input_condition=False,
        input_condition_mask=False
    ):
        super().__init__()
        assert not (type(self) == ResidualDiffusion and model.channels != model.out_dim)
        assert not model.random_or_learned_sinusoidal_cond

        self.textModel = LanguageModel()
        self.unet_model = model
        self.channels = self.unet_model.channels
        self.image_size = image_size
        self.objective = objective
        self.input_condition = input_condition
        self.input_condition_mask = input_condition_mask

        self.sum_scale = sum_scale if sum_scale else 0.01
        ddim_sampling_eta = 0.

        alphas = gen_coefficients(timesteps, schedule="decreased")
        alphas_cumsum = alphas.cumsum(dim=0).clip(0, 1)
        alphas_cumsum_prev = F.pad(alphas_cumsum[:-1], (1, 0), value=1.)
        betas2 = gen_coefficients(timesteps, schedule="increased", sum_scale=self.sum_scale)
        betas2_cumsum = betas2.cumsum(dim=0).clip(0, 1)
        betas_cumsum = torch.sqrt(betas2_cumsum)
        betas2_cumsum_prev = F.pad(betas2_cumsum[:-1], (1, 0), value=1.)
        posterior_variance = betas2*betas2_cumsum_prev/betas2_cumsum
        posterior_variance[0] = 0

        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters
        # default num sampling timesteps to number of timesteps at training
        self.sampling_timesteps = default(sampling_timesteps, timesteps)

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps #True
        self.ddim_sampling_eta = ddim_sampling_eta

        def register_buffer(name, val): 
            return self.register_buffer(name, val.to(torch.float32))

        register_buffer('alphas', alphas)
        register_buffer('alphas_cumsum', alphas_cumsum)
        register_buffer('one_minus_alphas_cumsum', 1 - alphas_cumsum)
        register_buffer('betas2', betas2)
        register_buffer('betas', torch.sqrt(betas2))
        register_buffer('betas2_cumsum', betas2_cumsum)
        register_buffer('betas_cumsum', betas_cumsum)
        register_buffer('posterior_mean_coef1', betas2_cumsum_prev / betas2_cumsum)
        register_buffer('posterior_mean_coef2', (betas2 * alphas_cumsum_prev - betas2_cumsum_prev * alphas) / betas2_cumsum)
        register_buffer('posterior_mean_coef3', betas2 / betas2_cumsum)
        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))

        self.posterior_mean_coef1[0] = 0
        self.posterior_mean_coef2[0] = 0
        self.posterior_mean_coef3[0] = 1
        self.one_minus_alphas_cumsum[-1] = 1e-6

    def predict_noise_from_res(self, x_t, t, x_input, pred_res):
        return (x_t-x_input-(extract(self.alphas_cumsum, t, x_t.shape)-1)* pred_res) / extract(self.betas_cumsum, t, x_t.shape)

    def predict_start_from_xinput_noise(self, x_t, t, x_input, noise):
        return (x_t-extract(self.alphas_cumsum, t, x_t.shape)*x_input - extract(self.betas_cumsum, t, x_t.shape) * noise) / extract(self.one_minus_alphas_cumsum, t, x_t.shape)

    def predict_start_from_res_noise(self, x_t, t, x_res, noise):
        return x_t-extract(self.alphas_cumsum, t, x_t.shape) * x_res - extract(self.betas_cumsum, t, x_t.shape) * noise

    def q_posterior_from_res_noise(self, x_res, noise, x_t, t):
        return x_t-extract(self.alphas, t, x_t.shape) * x_res - (extract(self.betas2, t, x_t.shape)/extract(self.betas_cumsum, t, x_t.shape)) * noise

    def q_posterior(self, pred_res, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_t +
            extract(self.posterior_mean_coef2, t, x_t.shape) * pred_res +
            extract(self.posterior_mean_coef3, t, x_t.shape) * x_start
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x_input, x, time, x_input_condition=None, text_embd = None, clip_denoised=True):
        if self.input_condition:
            x_in = torch.cat((x, x_input, x_input_condition), dim=1)
        else:
            x_in = torch.cat((x, x_input), dim=1)
            # x_in = x_input - x

        model_output = self.unet_model(x_in, time, text_embd)
        maybe_clip = partial(torch.clamp, min=-1.0, max=1.0) if clip_denoised else identity

        if self.objective == 'pred_res_noise':
            pred_res = model_output[0]
            pred_noise = model_output[1]
            pred_res = maybe_clip(pred_res)
            x_start = self.predict_start_from_res_noise(x, time, pred_res, pred_noise)
            x_start = maybe_clip(x_start)
        elif self.objective == 'pred_res_add_noise':
            pred_res = model_output[0]
            pred_noise = model_output[1] - model_output[0]
            pred_res = maybe_clip(pred_res)
            x_start = self.predict_start_from_res_noise(x, time, pred_res, pred_noise)
            x_start = maybe_clip(x_start)
        elif self.objective == 'pred_x0_noise':
            pred_res = x_input-model_output[0]
            pred_noise = model_output[1]
            pred_res = maybe_clip(pred_res)
            x_start = maybe_clip(model_output[0])
        elif self.objective == 'pred_x0_add_noise':
            x_start = model_output[0]
            pred_noise = model_output[1] - model_output[0]
            pred_res = x_input-x_start
            pred_res = maybe_clip(pred_res)
            x_start = maybe_clip(model_output[0])
        elif self.objective == "pred_noise":
            pred_noise = model_output[0]
            x_start = self.predict_start_from_xinput_noise(x, time, x_input, pred_noise)
            x_start = maybe_clip(x_start)
            pred_res = x_input - x_start
            pred_res = maybe_clip(pred_res)
        elif self.objective == "pred_res":
            pred_res = model_output[0]
            pred_res = maybe_clip(pred_res)
            pred_noise = self.predict_noise_from_res(x, time, x_input, pred_res)
            x_start = x_input - pred_res
            x_start = maybe_clip(x_start)

        return ModelResPrediction(pred_res, pred_noise, x_start)

    def p_mean_variance(self, x_input, x, t, x_input_condition=0):
        preds = self.model_predictions(x_input, x, t, x_input_condition)
        pred_res = preds.pred_res
        x_start = preds.pred_x_start
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(pred_res=pred_res, x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x_input, x, t: int, x_input_condition=0):
        # b, *_, device = *x.shape, x.device
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x_input, x=x, t=batched_times, x_input_condition=x_input_condition)
        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, x_input, shape, last=True):
        if self.input_condition:
            # x_input_condition = x_input[2] * x_input[3]
            x_input_condition = x_input[4]
        else:
            x_input_condition = 0
        x_input = x_input[1]
        device = self.betas.device

        img = x_input + math.sqrt(self.sum_scale) * torch.randn(shape, device=device)
        input_add_noise = img

        if not last:
            img_list = []

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img, x_start = self.p_sample(x_input, img, t, x_input_condition)

            if not last:
                img_list.append(img)

        if not last:
            img_list = [input_add_noise]+img_list
        else:
            img_list = [input_add_noise, img]
        return unnormalize_to_zero_to_one(img_list)

    @torch.no_grad()
    def ddim_sample(self, data, shape, last=True):
        if self.input_condition:
            # x_input_condition = data[2] * data[3]
            x_input_condition = data[4]
        else:
            x_input_condition = None
        x_input = data[1]
        text_embd = self.textModel(data[5]) #[1, 384]

        batch, device = shape[0], self.betas.device
        total_timesteps, sampling_timesteps, eta = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta
        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = x_input + math.sqrt(self.sum_scale) * torch.randn(shape, device=device)
        input_add_noise = img
        x_start = None

        if not last:
            img_list = []

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            preds = self.model_predictions(x_input, img, time_cond, x_input_condition, text_embd)

            pred_res = preds.pred_res
            pred_noise = preds.pred_noise
            x_start = preds.pred_x_start
            # visualize_tensor(pred_res, name='pred_res')
            # visualize_tensor(pred_noise, name = 'pred_noise')
            # visualize_tensor(x_start, name='pred_x_0')
            # exit(0)

            if time_next < 0:
                img = x_start
                if not last:
                    img_list.append(img)
                continue

            alpha_cumsum = self.alphas_cumsum[time]
            alpha_cumsum_next = self.alphas_cumsum[time_next]
            alpha = alpha_cumsum-alpha_cumsum_next

            betas2_cumsum = self.betas2_cumsum[time]
            betas2_cumsum_next = self.betas2_cumsum[time_next]
            betas2 = betas2_cumsum-betas2_cumsum_next
            betas = betas2.sqrt()
            betas_cumsum = self.betas_cumsum[time]
            betas_cumsum_next = self.betas_cumsum[time_next]
            sigma2 = eta * (betas2*betas2_cumsum_next/betas2_cumsum)
            sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum = (betas2_cumsum_next-sigma2).sqrt() / betas_cumsum

            noise = 0 if eta==0 else torch.randn_like(img)

            type = "use_pred_noise"
            if type == "use_pred_noise":
                img = img - alpha * pred_res - (betas_cumsum - (betas2_cumsum_next - sigma2).sqrt()) * pred_noise + sigma2.sqrt() * noise
            elif type == "use_x_start":
                img = sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum * img + \
                    (1 - sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum) * x_start + \
                    (alpha_cumsum_next-alpha_cumsum * sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum) * pred_res + \
                    sigma2.sqrt() * noise
            elif type == "special_eta_0":
                img = img - alpha * pred_res - (betas_cumsum - betas_cumsum_next) * pred_noise
            elif type == "special_eta_1":
                img = img - alpha * pred_res - betas2 / betas_cumsum * pred_noise + betas * betas2_cumsum_next.sqrt() / betas_cumsum * noise

            if not last:
                img_list.append(img)

        if last:
            img_list = [input_add_noise, img]
        else:
            img_list = [input_add_noise] + img_list
        return unnormalize_to_zero_to_one(img_list)

    @torch.no_grad()
    def sample(self, x_input, batch_size=16, last=True):
        sample_fn = self.ddim_sample if self.is_ddim_sampling else self.p_sample_loop
        x_input[1] = normalize_to_neg_one_to_one(x_input[1])
        batch_size, channels, h, w = x_input[1].shape
        size = (batch_size, channels, h, w)

        return sample_fn(x_input, size, last=last)

    def q_sample(self, x_start, x_res, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return x_start + extract(self.alphas_cumsum, t, x_start.shape) * x_res + extract(self.betas_cumsum, t, x_start.shape) * noise

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, data, t, noise=None):
        x_start = data[0] # gt = imgs[0], input = imgs[1]
        x_input = data[1]
        # x_alpha = data[2]
        # x_degra = data[3]
        x_mask = data[4]
        text_embd = self.textModel(data[5]) #[1, 384]

        if self.input_condition:
            # x_input_condition = x_alpha * x_degra
            x_input_condition = x_mask
        else:
            x_input_condition = None

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_res = x_input - x_start

        # noise sample
        xt = self.q_sample(x_start, x_res, t, noise=noise)

        # predict and take gradient step
        if self.input_condition:
            x_in = torch.cat((xt, x_input, x_input_condition), dim=1)
        else: #进这个
            # x_in = x_input - xt #[B, 3, 256, 256]
            x_in = torch.cat((xt, x_input), dim=1) #[B, 6, 256, 256]
            
        model_out = self.unet_model(x_in, t, text_embd) #[1, 3, 256, 256], [1, 3, 256, 256]

        target = []
        if self.objective == 'pred_res_noise':
            target.append(x_res)
            target.append(noise)
            pred_res = model_out[0]
            pred_noise = model_out[1]
        elif self.objective == 'pred_res_add_noise':
            target.append(x_res)
            target.append(x_res+noise)
            pred_res = model_out[0]
            pred_noise = model_out[1]-model_out[0]
        elif self.objective == 'pred_x0_noise':
            target.append(x_start)
            target.append(noise)
            pred_res = x_input-model_out[0]
            pred_noise = model_out[1]
        elif self.objective == 'pred_x0_add_noise':
            target.append(x_start)
            target.append(x_start+noise)
            pred_res = x_input-model_out[0]
            pred_noise = model_out[1] - model_out[0]
        elif self.objective == "pred_noise":
            target.append(noise)
            pred_noise = model_out[0]
        elif self.objective == "pred_res":
            target.append(x_res)
            pred_res = model_out[0]
        else:
            raise ValueError(f'unknown objective {self.objective}')

        u_loss = False
        if u_loss:
            x_u = self.q_posterior_from_res_noise(pred_res, pred_noise, xt, t)
            u_gt = self.q_posterior_from_res_noise(x_res, noise, xt, t)
            loss = 10000 * self.loss_fn(x_u, u_gt, reduction='none')
        else:
            loss = 0
            for i in range(len(model_out)):
                loss = loss + self.loss_fn(model_out[i], target[i], reduction='none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean') #[1, 196608]
        
        # fourier_res_penalty = abs(torch.fft.fft2(pred_res)).mean()

        # x_0_pred = self.predict_start_from_res_noise(xt, t, pred_res, pred_noise)
        # x_0_LR = tensor_singular_value_thresholding(x_start, tau=1.0) #tau=0.1
        # LR_loss = F.mse_loss(x_0_pred, x_0_LR)
        # visualize_tensor(x_start)
        # visualize_tensor(x_0_LR)
        # exit(0)

        return loss.mean() #+ 0.2 * LR_loss.mean()


    def forward(self, data, *args, **kwargs):
        assert isinstance(data, list)
        b, c, h, w, device, = *data[0].shape, data[0].device
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        # img = normalize_to_neg_one_to_one(img) #使得alpha不进行normalize
        data[0] = normalize_to_neg_one_to_one(data[0])
        data[1] = normalize_to_neg_one_to_one(data[1])

        return self.p_losses(data, t, *args, **kwargs)