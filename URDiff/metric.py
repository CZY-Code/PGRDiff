import os
import numpy as np
import cv2
from tqdm import tqdm
from pytorch_fid import fid_score
import lpips
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def calculate_lpips(loss_fn, img1, img2):
    img1 = torch.from_numpy(img1).float().permute(2, 0, 1).unsqueeze(0).cuda() / 255.0
    img2 = torch.from_numpy(img2).float().permute(2, 0, 1).unsqueeze(0).cuda() / 255.0
    distance = loss_fn(img1, img2)
    return distance.item()

def compare_images(folder_pred, folder_GT):
    """比较文件夹 folder_pred 和 folder_GT 中相同命名的图像"""
    test_PSNR = []
    test_SSIM = []
    test_LPIPS = []
    loss_fn = lpips.LPIPS(net='alex').cuda()
    for filename in tqdm(os.listdir(folder_pred)):
        path_pred = os.path.join(folder_pred, filename)
        path_GT = os.path.join(folder_GT, filename)
        if os.path.isfile(path_GT):
            img_a = cv2.imread(path_pred)
            img_b = cv2.imread(path_GT)

            if img_a is not None and img_b is not None:
                # 计算 PSNR
                test_PSNR.append(psnr(img_a, img_b))
                test_SSIM.append(ssim(img_a, img_b, channel_axis=2, data_range=255))
                test_LPIPS.append(calculate_lpips(loss_fn, img_a, img_b))
            else:
                print(f'Error reading images: {filename}')
        else:
            print(f'File not found in folder: {filename}')

    fid_value = calculate_fid(folder_pred, folder_GT, device='cuda')
    
    print(f"PSNR: {np.mean(test_PSNR):.2f}, SSIM: {np.mean(test_SSIM):.3f}, LPIPS: {np.mean(test_LPIPS):.3f}, FID: {fid_value*1000:.3f}")

def compare_images_V2(folder_pred, folder_GT, folder_Mask):
    test_metric = {'PSNR':[[],[],[],[]],
                   'SSIM':[[],[],[],[]],
                   'LPIPS':[[],[],[],[]],}
    loss_fn = lpips.LPIPS(net='alex').cuda() 

    for filename in tqdm(os.listdir(folder_pred)):
        path_pred = os.path.join(folder_pred, filename)
        path_GT = os.path.join(folder_GT, filename)
        path_mask = os.path.join(folder_Mask, filename)
        if os.path.isfile(path_GT):
            img_a = cv2.imread(path_pred)
            img_b = cv2.imread(path_GT)

            mask = cv2.imread(path_mask, cv2.IMREAD_GRAYSCALE)/255
            deg_ratio = mask.sum()/(mask.shape[0]*mask.shape[1])
            deg_ratio = int(deg_ratio*5)
            if deg_ratio > 2:
                deg_ratio = 2

            if img_a is not None and img_b is not None:
                psnr_img = psnr(img_a, img_b)
                ssim_img = ssim(img_a, img_b, channel_axis=2, data_range=255)
                lpips_img = calculate_lpips(loss_fn, img_a, img_b)
                test_metric['PSNR'][deg_ratio].append(psnr_img)
                test_metric['SSIM'][deg_ratio].append(ssim_img)
                test_metric['LPIPS'][deg_ratio].append(lpips_img)
                test_metric['PSNR'][-1].append(psnr_img)
                test_metric['SSIM'][-1].append(ssim_img)
                test_metric['LPIPS'][-1].append(lpips_img)
            else:
                print(f'Error reading images: {filename}')
        else:
            print(f'File not found in folder: {filename}')

    fid_value = calculate_fid(folder_pred, folder_GT, device='cuda')
    for i in range(3):
        print(f"Deg Ratio: {i}, PSNR: {np.mean(test_metric['PSNR'][i]):.2f}, SSIM: {np.mean(test_metric['SSIM'][i]):.3f}, LPIPS: {np.mean(test_metric['LPIPS'][i]):.3f}")

    print(f"Total metric: PSNR: {np.mean(test_metric['PSNR'][-1]):.2f}, SSIM: {np.mean(test_metric['SSIM'][-1]):.3f}, LPIPS: {np.mean(test_metric['LPIPS'][-1]):.3f}, FID: {fid_value*1000:.3f}")

def calculate_fid(folder_pred, folder_GT, device='cpu'):
    fid_value = fid_score.calculate_fid_given_paths(
        [folder_pred, folder_GT],
        batch_size=64,
        device=device,
        dims=64,
        num_workers=4
    )
    return fid_value


if __name__ == '__main__':
    folder_pred = './results/Ours-DH/'
    folder_GT = './results/GT-DH/'
    folder_Mask = './results/Mask-DH/'

    # folder_pred = './results/Ours-TEN/'
    # folder_GT = './results/GT-TEN/'
    # folder_Mask = './results/Mask-TEN/'

    # compare_images(folder_pred, folder_GT)
    compare_images_V2(folder_pred, folder_GT, folder_Mask)
