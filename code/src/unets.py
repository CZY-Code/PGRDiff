import torch
from torch import nn
from .modules import *
from .ICB import ICB

class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        resnet_block_groups=8,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16,
        input_condition=False
    ):
        super().__init__()
        # determine dimensions
        self.channels = channels
        input_channels = 2 * channels + 1 * (1 if input_condition else 0)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:])) #[(64, 64), (64, 128), (128, 256), (256, 512)]
        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        time_dim = dim * 4
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(
                learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # for ind, (dim_in, dim_out) in enumerate(in_out):
        #     is_last = ind >= (num_resolutions - 1)
        #     self.downs.append(nn.ModuleList([
        #         block_klass(dim_in, dim_in, time_emb_dim=time_dim),
        #         block_klass(dim_in, dim_in, time_emb_dim=time_dim),
        #         Residual(PreNorm(dim_in, LinearAttention(dim_in))),
        #         Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(
        #             dim_in, dim_out, 3, padding=1)
        #     ]))
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                ICB(feature_dim=dim_in, text_dim=384), #TODO
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(
                    dim_in, dim_out, 3, padding=1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(
                    dim_out, dim_in, 3, padding=1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)
        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time, textemb):
        x = self.init_conv(x)
        r = x.clone()
        t = self.time_mlp(time)
        h = []

        for block1, icb, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)
            x = icb(x, textemb)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        return self.final_conv(x)


class UnetRes(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        resnet_block_groups=8,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16,
        share_encoder=1,
        input_condition=False
    ):
        super().__init__()
        self.input_condition = input_condition
        self.share_encoder = share_encoder
        self.channels = channels
        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        # determine dimensions
        if self.share_encoder == 1:
            input_channels = 2 * channels + channels * (1 if input_condition else 0)
            init_dim = default(init_dim, dim)
            self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)

            dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
            in_out = list(zip(dims[:-1], dims[1:]))
            block_klass = partial(ResnetBlock, groups=resnet_block_groups)
            # time embeddings
            time_dim = dim * 4

            if self.random_or_learned_sinusoidal_cond:
                sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(
                    learned_sinusoidal_dim, random_fourier_features)
                fourier_dim = learned_sinusoidal_dim + 1
            else:
                sinu_pos_emb = SinusoidalPosEmb(dim)
                fourier_dim = dim

            self.time_mlp = nn.Sequential(
                sinu_pos_emb,
                nn.Linear(fourier_dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim)
            )

            # layers
            self.downs = nn.ModuleList([])
            self.ups = nn.ModuleList([])
            self.ups_no_skip = nn.ModuleList([])
            num_resolutions = len(in_out)

            for ind, (dim_in, dim_out) in enumerate(in_out):
                is_last = ind >= (num_resolutions - 1)

                self.downs.append(nn.ModuleList([
                    block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                    block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                    Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                    Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1)
                ]))

            mid_dim = dims[-1]
            self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
            self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
            self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

            for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
                is_last = ind == (len(in_out) - 1)

                self.ups.append(nn.ModuleList([
                    block_klass(dim_out + dim_in, dim_out,time_emb_dim=time_dim),
                    block_klass(dim_out + dim_in, dim_out,time_emb_dim=time_dim),
                    Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                    Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1)
                ]))

                self.ups_no_skip.append(nn.ModuleList([
                    block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                    block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                    Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                    Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1)
                ]))

            self.final_res_block_1 = block_klass(dim, dim, time_emb_dim=time_dim)
            self.final_conv_1 = nn.Conv2d(dim, self.out_dim, 1)

            self.final_res_block_2 = block_klass(dim * 2, dim, time_emb_dim=time_dim)
            self.final_conv_2 = nn.Conv2d(dim, self.out_dim, 1)

        elif self.share_encoder == 0:
            self.unet0 = Unet(dim,
                              init_dim=init_dim,
                              out_dim=out_dim,
                              dim_mults=dim_mults,
                              channels=channels,
                              resnet_block_groups=resnet_block_groups,
                              learned_variance=learned_variance,
                              learned_sinusoidal_cond=learned_sinusoidal_cond,
                              random_fourier_features=random_fourier_features,
                              learned_sinusoidal_dim=learned_sinusoidal_dim,
                              input_condition=input_condition)
            self.unet1 = Unet(dim,
                              init_dim=init_dim,
                              out_dim=out_dim,
                              dim_mults=dim_mults,
                              channels=channels,
                              resnet_block_groups=resnet_block_groups,
                              learned_variance=learned_variance,
                              learned_sinusoidal_cond=learned_sinusoidal_cond,
                              random_fourier_features=random_fourier_features,
                              learned_sinusoidal_dim=learned_sinusoidal_dim,
                              input_condition=input_condition)
            
        elif self.share_encoder == -1:
            self.unet0 = Unet(dim,
                              init_dim=init_dim,
                              out_dim=out_dim,
                              dim_mults=dim_mults,
                              channels=channels,
                              resnet_block_groups=resnet_block_groups,
                              learned_variance=learned_variance,
                              learned_sinusoidal_cond=learned_sinusoidal_cond,
                              random_fourier_features=random_fourier_features,
                              learned_sinusoidal_dim=learned_sinusoidal_dim,
                              input_condition=input_condition)

    def forward(self, x, time, textemb):
        if self.share_encoder == 0: #TODO 感觉unet1预测纯噪声不用加文本进去
            return self.unet0(x, time, textemb), self.unet1(x, time, textemb)
        
        elif self.share_encoder == -1:
            return [self.unet0(x, time)]
        
        elif self.share_encoder == 1:
            x = self.init_conv(x)
            r = x.clone()
            t = self.time_mlp(time)

            h = []
            for block1, block2, attn, downsample in self.downs:
                x = block1(x, t)
                h.append(x)
                x = block2(x, t)
                x = attn(x)
                h.append(x)
                x = downsample(x)

            x = self.mid_block1(x, t)
            x = self.mid_attn(x)
            x = self.mid_block2(x, t)
            out_res = x
            for block1, block2, attn, upsample in self.ups_no_skip:
                out_res = block1(out_res, t)
                out_res = block2(out_res, t)
                out_res = attn(out_res)
                out_res = upsample(out_res)

            out_res = self.final_res_block_1(out_res, t)
            out_res = self.final_conv_1(out_res)

            for block1, block2, attn, upsample in self.ups:
                x = torch.cat((x, h.pop()), dim=1)
                x = block1(x, t)

                x = torch.cat((x, h.pop()), dim=1)
                x = block2(x, t)
                x = attn(x)

                x = upsample(x)

            x = torch.cat((x, r), dim=1)
            x = self.final_res_block_2(x, t)
            out_res_add_noise = self.final_conv_2(x)

            return out_res, out_res_add_noise
