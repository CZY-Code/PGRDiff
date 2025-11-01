import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

from src.residual_diff import ResidualDiffusion
from src.unets import UnetRes
from src.residual_denoising_diffusion_pytorch import Trainer
from src.utils import set_seed
from metric import compare_images

# init 
set_seed(3407)
save_and_sample_every = 2000 #1000
sampling_timesteps = 10 #20
timesteps = 100 #100
train_num_steps = 100000 #50000

input_condition = True
input_condition_mask = False
is_dunhuang = True
if is_dunhuang:
    folder = ["/home/chengzy/PGRDiff/DUNHUANG/train/train_GT", "/home/chengzy/PGRDiff/muralv2/masks"]
else:
    folder = ["/home/chengzy/PGRDiff/muralv2/images", "/home/chengzy/PGRDiff/muralv2/masks"]

train_batch_size = 1 #TODO
num_samples = 1
sum_scale = 1
image_size = 256

model = UnetRes(
    dim=64,
    dim_mults=(1, 2, 4, 8),
    share_encoder=0,
    input_condition=input_condition
)
diffusion = ResidualDiffusion(
    model,
    image_size=image_size,
    timesteps=timesteps,      #1000 number of steps
    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    sampling_timesteps=sampling_timesteps,
    objective='pred_res_noise',
    loss_type='l2',            # L1 or L2
    sum_scale = sum_scale,
    input_condition=input_condition,
    input_condition_mask=input_condition_mask
)

trainer = Trainer(
    diffusion,
    folder,
    train_batch_size=train_batch_size,
    num_samples=num_samples,
    train_lr=1e-4, #8e-5
    train_num_steps=train_num_steps,         # total training steps
    gradient_accumulate_every=4,    # gradient accumulation steps
    ema_decay=0.995,                # exponential moving average decay
    amp=False,                        # turn on mixed precision
    convert_image_to="RGB",
    save_and_sample_every=save_and_sample_every,
    equalizeHist=False,
    crop_patch=True,
    is_dunhuang=is_dunhuang,
)

# if trainer.accelerator.is_local_main_process:
#     trainer.load('50')
    
# train
trainer.train()

# test
if trainer.accelerator.is_local_main_process:
    trainer.load('50-PG-512-DH')
    trainer.set_results_folder('./results/test_timestep_'+str(sampling_timesteps))
    trainer.test(last=False)
    if is_dunhuang:
        compare_images('./results/Ours-DH/', './results/GT-DH/')
    else:
        compare_images('./results/Ours-TEN/', './results/GT-TEN/')
