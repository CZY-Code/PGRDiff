import math
import os
from pathlib import Path
import torch

from accelerate import Accelerator
from datasets.get_dataset import dataset
from ema_pytorch import EMA
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import utils
from tqdm.auto import tqdm

from .utils import *

# trainer class
class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        folder,
        *,
        train_batch_size=16,
        gradient_accumulate_every=1,
        augment_flip=True,
        train_lr=1e-4,
        train_num_steps=100000,
        ema_update_every=10,
        ema_decay=0.995,
        adam_betas=(0.9, 0.99),
        save_and_sample_every=1000,
        num_samples=25,
        results_folder='./results/sample',
        amp=False,
        fp16=False,
        convert_image_to=None,
        sub_dir=False,
        equalizeHist=False,
        crop_patch=False,
        is_dunhuang=True
    ):
        super().__init__()
        self.accelerator = Accelerator(mixed_precision='fp16' if fp16 else 'no', 
                                       gradient_accumulation_steps=gradient_accumulate_every)
        self.sub_dir = sub_dir
        self.crop_patch = crop_patch
        self.accelerator.native_amp = amp
        self.model = diffusion_model

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        # image + mask
        self.sample_dataset = dataset(folder, self.image_size, augment_flip=False, convert_image_to=convert_image_to, 
                                      equalizeHist=equalizeHist, crop_patch=crop_patch, sample=True, is_dunhuang=is_dunhuang)
        self.sample_loader = cycle(DataLoader(self.sample_dataset, batch_size=num_samples, shuffle=True,
                                                                        pin_memory=True, num_workers=4))

        train_ds = dataset(folder, self.image_size, augment_flip=augment_flip, convert_image_to=convert_image_to, 
                           equalizeHist=equalizeHist, crop_patch=crop_patch, is_dunhuang=is_dunhuang)
        self.training_dataloader = self.accelerator.prepare(DataLoader(train_ds, batch_size=train_batch_size, shuffle=True, 
                                                                        pin_memory=True, num_workers=4))

        # optimizer
        self.opt = Adam(diffusion_model.parameters(), lr=train_lr, betas=adam_betas)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, self.train_num_steps)
        # for logging results in a folder periodically
        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)
            self.ema.to(self.accelerator.device)
            self.set_results_folder(results_folder)

        # step counter state
        self.step = 0
        # prepare model, dataloader, optimizer with accelerator
        self.model, self.opt, self.lr_scheduler = self.accelerator.prepare(self.model, self.opt, self.lr_scheduler)
        self.device = self.accelerator.device

    def save(self, milestone):
        if self.accelerator.is_local_main_process:
            data = {
                'step': self.step,
                'model': self.accelerator.get_state_dict(self.model),
                'opt': self.opt.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict(),
                'ema': self.ema.state_dict(),
                'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
            }
            torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        path = Path(self.results_folder / f'model-{milestone}.pt') #BUG
        if path.exists():
            data = torch.load(str(path), map_location=self.device)
            model = self.accelerator.unwrap_model(self.model)
            model.load_state_dict(data['model'])
            self.step = data['step']
            self.opt.load_state_dict(data['opt'])
            self.lr_scheduler.load_state_dict(data['lr_scheduler'])
            self.ema.load_state_dict(data['ema'])
            if exists(self.accelerator.scaler) and exists(data['scaler']):
                self.accelerator.scaler.load_state_dict(data['scaler'])
            print("load model - "+str(path))
        # self.ema.to(self.device)

    def train(self):
        with tqdm(initial=self.step, total=self.train_num_steps, disable=not self.accelerator.is_main_process) as pbar:
            while self.step < self.train_num_steps:
                for data in self.training_dataloader:
                    with self.accelerator.accumulate(self.model):
                        # with self.accelerator.autocast():
                        loss = self.model(data)
                        self.accelerator.backward(loss)
                        self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.opt.step()
                        self.lr_scheduler.step()
                        self.opt.zero_grad()
                        self.step += 1
                        
                        self.accelerator.wait_for_everyone()
                        if self.accelerator.is_main_process:
                            self.ema.update()
                            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                                milestone = self.step // self.save_and_sample_every
                                self.sample(milestone)
                                if self.step % (self.save_and_sample_every * 10) == 0:
                                    self.save(milestone)

                    pbar.set_description(f'loss: {loss.item():.4f}')
                    pbar.update(1)

        self.accelerator.print('training complete')
    
    @torch.no_grad()
    def sample(self, milestone, last=True):
        self.ema.ema_model.eval()
        data = next(self.sample_loader)
        x_input_sample = [item.to(self.device) for item in data[:-1]]
        x_input_sample.append(data[-1]) # text不需要to(device)
        show_x_input_sample = x_input_sample[:2] #可视化img0、img1

        all_images_list = show_x_input_sample + list(self.ema.ema_model.sample(x_input_sample, batch_size=self.num_samples))
        all_images = torch.cat(all_images_list, dim=0)

        if last:
            nrow = int(math.sqrt(self.num_samples))
        else:
            nrow = all_images.shape[0]
            
        file_name = f'sample-{milestone}.png'
        utils.save_image(all_images, str(self.results_folder / file_name), nrow=nrow)
        print("sampe-save "+file_name)
        return milestone

    def test(self, last=True):
        print("test start")
        self.ema.ema_model.eval()
        loader = DataLoader(dataset=self.sample_dataset, batch_size=1)
        i = 0
        save_image_path = [
            '1-dunhuang-beiliang&beiwei_img194.jpg',
            '2-xiwei_img234.jpg',
            '3-dunhuang-beizhou_img122.jpg',
            '4-suidai_img129.1.jpg',
            '5-dunhuang-chutang_img146.jpg',
            '6-shengtang_img155.jpg',
            '7-zhongtang_img349.jpg',
            '8-wantang_img143.1.jpg',
            '9-dunhuang-wudai&song_img78.jpg',
            '10-dunhuang-xixia&yuan_img305.jpg'
        ]
        for data in loader:
            file_name = self.sample_dataset.load_name(i, sub_dir=self.sub_dir)
            i += 1
            # if file_name in save_image_path:
            with torch.no_grad():
                x_input_sample = [item.to(self.device) for item in data[:-1]]
                x_input_sample.append(data[-1]) # text不需要to(device)
                show_x_input_sample = x_input_sample[:2]
                all_images_list = show_x_input_sample + list(self.ema.ema_model.sample(x_input_sample, batch_size=self.num_samples, last=last))

            all_images = torch.cat(all_images_list, dim=0) #[4, 3, 256, 256] [23, 3, 256, 256]
            if last:
                nrow = int(math.sqrt(self.num_samples))
            else:
                nrow = all_images.shape[0]
            # utils.save_image(all_images, str(self.results_folder / file_name), nrow=nrow)
            
            # with open('./results/prompt.txt', 'a+', encoding='utf-8') as f:
            #     f.write(file_name+' '+ str(data[-1])+'\n')
            utils.save_image(all_images_list[-1], str('./results/Ours-DH/' + file_name))
            utils.save_image(all_images_list[0], str('./results/GT-DH/' + file_name))
            utils.save_image(all_images_list[1], str('./results/Input-DH/' + file_name))

            print("test-save " + file_name)
        print("test end")

    def set_results_folder(self, path):
        self.results_folder = Path(path)
        if not self.results_folder.exists():
            os.makedirs(self.results_folder)
