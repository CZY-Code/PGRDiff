import os
import random
from pathlib import Path
import torch
import Augmentor
import cv2
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

class DunhangDataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts=['jpg', 'jpeg', 'png', 'tiff'],
        augment_flip=False,
        convert_image_to=None,
        equalizeHist=False,
        crop_patch=True,
        sample=False
    ):
        super().__init__()
        self.equalizeHist = equalizeHist
        self.exts = exts
        self.augment_flip = augment_flip
        self.crop_patch = crop_patch
        self.sample = sample

        self.image_list = self.load_flist(folder[0])
        self.image_list.sort()
        self.masks = self.load_flist(folder[1])
        train_num = int(len(self.image_list)*0.8)
        if self.sample: #test
            self.gt = self.image_list[train_num:]
        else: #train
            self.gt = self.image_list[:train_num]
            

        self.image_size = image_size
        self.convert_image_to = convert_image_to

        self.damage_type_dict = {
            "CausticSodaCrystallization": "caustic soda crystallization",
            "ChangeColor": "color change",
            "Crack": "cracking",
            "Fading": "fading",
            "FallenOff": "detachment",
            "InsectInfestation": "insect infestation",
            "Moldy": "mold growth",
            "Scratch": "scratching",
            "SmokeSmoke": "smoke damage",
            "WaterStains": "water staining"
        }
        self.damage_level = ['mild', 'moderate', 'serious']

    def __len__(self):
        return len(self.gt)

    def __getitem__(self, index):
        img0 = Image.open(self.gt[index])
        # img0 = Image.open('/home/chengzy/PGRDiff/muralv2/images/9-dunhuang-wudai&song/img146_2.jpg')
        img0 = convert_image_to_fn(self.convert_image_to, img0) if self.convert_image_to else img0
        # img0 = self.pad_img(img0, self.image_size) #转numpy
        img0 = np.asarray(img0) #转numpy
        if self.crop_patch:
            img0 = self.get_patch(img0, 512)
        img0 = self.cv2equalizeHist(img0) if self.equalizeHist else img0
        selected_mask = random.choice(self.masks)
        # selected_mask = Path(os.path.abspath('/home/chengzy/PGRDiff/muralv2/masks/Crack/18.png'))
        damage_type = self.damage_type_dict.get(selected_mask.parts[6], "general damage")
        mask = Image.open(selected_mask)
        # mask = Image.open(self.masks[index % len(self.masks)])
        mask = convert_image_to_fn('RGBA', mask) if self.convert_image_to else mask
        mask = np.asarray(mask) #[600, 600, 4]
        
        images = [[img0, mask]]
        p = Augmentor.DataPipeline(images)
        if self.augment_flip:
            p.flip_left_right(0.5) #1
        p.resize(1, self.image_size, self.image_size)
        g = p.generator(batch_size=1)
        augmented_images = next(g) #一次生成
        img0 = augmented_images[0][0]  # (H,W,3)
        mask_rgba = augmented_images[0][1]

        deg_rgb = mask_rgba[..., :3]  # (H,W,3) #[0,1]
        alpha = mask_rgba[..., 3:] / 255.0  # (H,W,1)
        # deg_rgb = np.where(deg_rgb>0, 255, 0).astype(np.uint8) #
        # alpha = np.where(alpha>=0.5, 1.0, alpha) #TODO
        img1 = (deg_rgb * alpha + img0 * (1 - alpha)).astype(np.uint8)
        
        # cv2.imwrite("img0.jpg", cv2.cvtColor(img0, cv2.COLOR_RGB2BGR))  # 保存图像
        # cv2.imwrite("img1.jpg", cv2.cvtColor(img1, cv2.COLOR_RGB2BGR))  # 保存图像
        # cv2.imwrite("deg.jpg", cv2.cvtColor(np.where(deg_rgb==0, 255, deg_rgb), cv2.COLOR_RGB2BGR))  # 保存图像
        # cv2.imwrite("alpha.jpg", (1 - alpha) * 255)  # 保存图像
        # cv2.imwrite("mask.jpg", np.where(alpha > 0, 0, 255).astype(np.uint8))  # 保存图像
        # cv2.imwrite("res.jpg", cv2.cvtColor(img1-img0, cv2.COLOR_RGB2BGR))  # 保存图像
        # exit(0)

        alpha = torch.from_numpy(alpha.transpose((2, 0, 1))).contiguous().float()
        mask = torch.where(alpha > 0, torch.ones_like(alpha), torch.zeros_like(alpha))
        damage_level = self.damage_level[int(torch.sum(alpha) / torch.sum(mask) * len(self.damage_level))]
        # 定义变量
        dynasty = "Tang"  # 朝代
        location = "Dunhuang Mogao Grottoes"  # 地点
        prompt = f"Please restore the mural from the {dynasty} period located in {location}, focusing on addressing the {damage_level} {damage_type} issues."
        return [self.to_tensor(img0), self.to_tensor(img1), 
                alpha, self.to_tensor(deg_rgb), 
                mask, prompt]

    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist): #递归遍历当前文件夹及其所有子文件夹
                return [p for ext in self.exts for p in Path(f'{flist}').glob(f'**/*.{ext}')]

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                except:
                    return [flist]
        return []

    def cv2equalizeHist(self, img):
        (r, g, b) = cv2.split(img)
        r = cv2.equalizeHist(r)
        g = cv2.equalizeHist(g)
        b = cv2.equalizeHist(b)
        img = cv2.merge((r, g, b))
        return img

    def to_tensor(self, img):
        img = Image.fromarray(img)  # returns an image object.
        img_t = TF.to_tensor(img).float()
        return img_t

    def load_name(self, index, sub_dir=False):
        name = self.gt[index]
        if sub_dir == 0:
            return os.path.basename(name)
        elif sub_dir == 1:
            path = os.path.dirname(name)
            sub_dir = (path.split("/"))[-1]
            return sub_dir+"_"+os.path.basename(name)

    def get_patch(self, img, patch_size):
        h, w = img.shape[:2]
        if h<=patch_size or w<=patch_size:
            return img
        rr = random.randint(0, h-patch_size)
        cc = random.randint(0, w-patch_size)
        img = img[rr:rr+patch_size, cc:cc+patch_size, :]
        return img

    def pad_img(self, img, patch_size, block_size=8):
        # img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        img = np.asarray(img)
        h, w = img.shape[:2]
        bottom = 0
        right = 0
        if h < patch_size:
            bottom = patch_size-h
            h = patch_size
        if w < patch_size:
            right = patch_size-w
            w = patch_size
        bottom = bottom + (h // block_size) * block_size + (block_size if h % block_size != 0 else 0) - h
        right = right + (w // block_size) * block_size + (block_size if w % block_size != 0 else 0) - w
        img= cv2.copyMakeBorder(img, 0, bottom, 0, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return img


class MuralDataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts=['jpg', 'jpeg', 'png', 'tiff'],
        augment_flip=False,
        convert_image_to=None,
        equalizeHist=False,
        crop_patch=True,
        sample=False
    ):
        super().__init__()
        self.equalizeHist = equalizeHist
        self.exts = exts
        self.augment_flip = augment_flip
        self.crop_patch = crop_patch
        self.sample = sample
        self.test_path = []
        self.image_list = self.load_flist(folder[0])
        random.shuffle(self.image_list)
        self.masks = self.load_flist(folder[1])
        train_num = int(len(self.image_list)*0.9)
        if self.sample: #test
            self.gt = self.image_list[train_num:]
        else: #train
            self.gt = self.image_list[:train_num]
            

        self.image_size = image_size
        self.convert_image_to = convert_image_to

        self.damage_type_dict = {
            "CausticSodaCrystallization": "caustic soda crystallization",
            "ChangeColor": "color change",
            "Crack": "cracking",
            "Fading": "fading",
            "FallenOff": "detachment",
            "InsectInfestation": "insect infestation",
            "Moldy": "mold growth",
            "Scratch": "scratching",
            "SmokeSmoke": "smoke damage",
            "WaterStains": "water staining"
        }
        self.damage_level = ['mild', 'moderate', 'serious']
        self.dynasty_dict = {
            "1-dunhuang-beiliang&beiwei": "Northern Liang & Northern Wei",
            "2-xiwei": "Western Wei",
            "3-dunhuang-beizhou": "Northern Zhou",
            "4-suidai": "Sui Dynasty",
            "5-dunhuang-chutang": "Early Tang",
            "6-shengtang": "High Tang",
            "7-zhongtang": "Middle Tang",
            "8-wantang": "Late Tang",
            "9-dunhuang-wudai&song": "Five Dynasties & Song Dynasty"
        }

    def __len__(self):
        return len(self.gt)

    def __getitem__(self, index):
        img0 = Image.open(self.gt[index])
        img0 = convert_image_to_fn(self.convert_image_to, img0) if self.convert_image_to else img0
        img0 = np.asarray(img0) #转numpy
        if self.crop_patch:
            img0 = self.get_patch(img0, 512)
        img0 = self.cv2equalizeHist(img0) if self.equalizeHist else img0
        selected_mask = random.choice(self.masks)
        damage_type = self.damage_type_dict.get(selected_mask.parts[6], "general damage")
        mask = Image.open(selected_mask)
        mask = convert_image_to_fn('RGBA', mask) if self.convert_image_to else mask
        mask = np.asarray(mask) #[600, 600, 4]
        
        images = [[img0, mask]]
        p = Augmentor.DataPipeline(images)
        if self.augment_flip:
            p.flip_left_right(0.5) #1
        p.resize(1, self.image_size, self.image_size)
        g = p.generator(batch_size=1)
        augmented_images = next(g) #一次生成
        img0 = augmented_images[0][0]  # (H,W,3)
        mask_rgba = augmented_images[0][1]

        deg_rgb = mask_rgba[..., :3]  # (H,W,3) #[0,1]
        alpha = mask_rgba[..., 3:] / 255.0  # (H,W,1)，扩展维度便于广播
        # deg_rgb = np.where(deg_rgb > 0, 255, 0).astype(np.uint8) #TODO
        # alpha = np.where(alpha > 0.5, 0.8, alpha) #TODO
        img1 = (deg_rgb * alpha + img0 * (1 - alpha)).astype(np.uint8)
        
        # cv2.imwrite("img0.jpg", cv2.cvtColor(img0, cv2.COLOR_RGB2BGR))  # 保存图像
        # cv2.imwrite("img1.jpg", cv2.cvtColor(img1, cv2.COLOR_RGB2BGR))  # 保存图像
        # cv2.imwrite("deg.jpg", cv2.cvtColor(np.where(deg_rgb==0, 255, deg_rgb), cv2.COLOR_RGB2BGR))  # 保存图像
        # cv2.imwrite("alpha.jpg", (1 - alpha) * 255)  # 保存图像
        # cv2.imwrite("mask.jpg", np.where(alpha > 0, 0, 255).astype(np.uint8))  # 保存图像
        # cv2.imwrite("res.jpg", cv2.cvtColor(img1-img0, cv2.COLOR_RGB2BGR))  # 保存图像
        # exit(0)

        alpha = torch.from_numpy(alpha.transpose((2, 0, 1))).contiguous().float()
        mask = torch.where(alpha > 0, torch.ones_like(alpha), torch.zeros_like(alpha))
        damage_level = self.damage_level[int(torch.sum(alpha) / torch.sum(mask) * len(self.damage_level))]
        # damage_level = 'serious' #TODO
        # 定义变量
        dynasty = self.dynasty_dict.get(self.gt[index].parts[6], "Tang")
        location = "Dunhuang Mogao Grottoes"  # 地点
        prompt = f"Please restore the mural from the {dynasty} period located in {location}, focusing on addressing the {damage_level} {damage_type} issues."
        return [self.to_tensor(img0), self.to_tensor(img1), 
                alpha, self.to_tensor(deg_rgb), 
                mask, prompt]

    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist): #递归遍历当前文件夹及其所有子文件夹
                return [p for ext in self.exts for p in Path(f'{flist}').glob(f'**/*.{ext}')]

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                except:
                    return [flist]
        return []

    def cv2equalizeHist(self, img):
        (r, g, b) = cv2.split(img)
        r = cv2.equalizeHist(r)
        g = cv2.equalizeHist(g)
        b = cv2.equalizeHist(b)
        img = cv2.merge((r, g, b))
        return img

    def to_tensor(self, img):
        img = Image.fromarray(img)  # returns an image object.
        img_t = TF.to_tensor(img).float()
        return img_t

    def load_name(self, index, sub_dir=False):
        name = self.gt[index]
        path = os.path.dirname(name)
        sub_dir = (path.split("/"))[-1]
        return sub_dir+"_"+os.path.basename(name)

    def get_patch(self, img, patch_size):
        h, w = img.shape[:2]
        if h<=patch_size or w<=patch_size:
            return img
        rr = random.randint(0, h-patch_size)
        cc = random.randint(0, w-patch_size)
        img = img[rr:rr+patch_size, cc:cc+patch_size, :]
        return img

    def pad_img(self, img, patch_size, block_size=8):
        # img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        img = np.asarray(img)
        h, w = img.shape[:2]
        bottom = 0
        right = 0
        if h < patch_size:
            bottom = patch_size-h
            h = patch_size
        if w < patch_size:
            right = patch_size-w
            w = patch_size
        bottom = bottom + (h // block_size) * block_size + (block_size if h % block_size != 0 else 0) - h
        right = right + (w // block_size) * block_size + (block_size if w % block_size != 0 else 0) - w
        img= cv2.copyMakeBorder(img, 0, bottom, 0, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return img