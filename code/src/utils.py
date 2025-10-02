import math
import random
import numpy as np
import torch
import cv2

def set_seed(SEED):
    # initialize random seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

# normalization functions
def normalize_to_neg_one_to_one(img):
    if isinstance(img, list):
        return [img[k] * 2 - 1 for k in range(len(img))]
    else:
        return img * 2 - 1

def unnormalize_to_zero_to_one(img):
    if isinstance(img, list):
        return [(img[k] + 1) * 0.5 for k in range(len(img))]
    else:
        return (img + 1) * 0.5
    
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def gen_coefficients(timesteps, schedule="increased", sum_scale=1):
    if schedule == "increased":
        x = torch.linspace(1, timesteps, timesteps, dtype=torch.float64)
        scale = 0.5*timesteps*(timesteps+1)
        alphas = x/scale
    elif schedule == "decreased":
        x = torch.linspace(1, timesteps, timesteps, dtype=torch.float64)
        x = torch.flip(x, dims=[0])
        scale = 0.5*timesteps*(timesteps+1)
        alphas = x/scale
    elif schedule == "average":
        alphas = torch.full([timesteps], 1/timesteps, dtype=torch.float64)
    else:
        alphas = torch.full([timesteps], 1/timesteps, dtype=torch.float64)
    assert alphas.sum()-torch.tensor(1) < torch.tensor(1e-10)

    return alphas*sum_scale

@torch.no_grad()
def tensor_singular_value_thresholding(x, tau):
    """
    对[B, C, H, W]形状的张量在通道维度(C)上执行奇异值阈值算法
    对每个批次样本单独处理，在H×W的空间切片上执行SVD
    
    参数:
        x: 输入张量，形状为[B, C, H, W]
        tau: 阈值参数，tau > 0
        
    返回:
        处理后的张量，形状与输入相同[B, C, H, W]
    """
    B, C, H, W = x.shape
    # 步骤1: 在通道维度执行傅里叶变换
    x_fft = torch.fft.fft(x, dim=1)  # 结果为复数张量
    # 步骤2: 计算前半部分通道数（显式向上取整）
    half_c = torch.ceil(torch.tensor((C + 1) / 2)).long().item()
    w = torch.zeros_like(x_fft, dtype=torch.complex64)
    
    # 处理前半部分通道的频谱分量
    for b in range(B):
        # 对每个批次样本单独处理
        for c in range(half_c):
            # 提取单个样本的空间切片 [H, W]
            spatial_slice = x_fft[b, c, :, :]
            # 在H×W的空间切片上执行SVD
            U, S, Vh = torch.linalg.svd(spatial_slice) #为啥用torch.svd会出现通道颜色不对的问题？？？
            # 应用阈值函数 (S - τ)_+
            S_thresh = torch.max(S - tau, torch.zeros_like(S))
            # 重构处理后的空间切片
            bar_w = U @ torch.diag(S_thresh).to(torch.complex64) @ Vh.t()
            # 将处理结果赋值回张量
            w[b, c, :, :] = bar_w

    # 利用共轭对称性处理后半部分通道
    for c in range(half_c, C):  
        w[:, c, :, :] = torch.conj(w[:, C - c, :, :]) # 对称索引

    # 步骤3: 逆傅里叶变换并取实部
    result = torch.fft.ifft(w, dim=1).real
    return result


def visualize_tensor(x_0_LR, normalize=True, index=0, wait_time=0, name='test'):
    """
    可视化[B, C, H, W]形状的图像张量
    
    参数:
        x_0_LR: 输入张量，形状为[B, C, H, W]
        normalize: 是否需要归一化（如果张量值不在0-255范围）
        index: 要可视化的批次中的索引（默认第0个）
        wait_time: OpenCV窗口等待时间（0表示手动关闭，单位ms）
    """
    # 1. 选择批次中的一个样本（默认第0个）
    x = x_0_LR[index]  # 形状变为[C, H, W]
    
    # 2. 转换为numpy数组并调整通道顺序（C, H, W -> H, W, C）
    x_np = x.detach().cpu().numpy()  # 从GPU移到CPU并转为numpy
    x_np = np.transpose(x_np, (1, 2, 0))  # 通道维度移到最后
    
    # 3. 处理通道数（如果是单通道灰度图，转为3通道以兼容OpenCV显示）
    if x_np.shape[-1] == 1:
        x_np = np.repeat(x_np, 3, axis=-1)  # 单通道转3通道
    
    # 4. 归一化到0-255范围（如果需要）
    if normalize:
        x_np = (x_np - x_np.min()) / (x_np.max() - x_np.min() + 1e-8)  # 归一化到0-1
        x_np = (x_np * 255).astype(np.uint8)  # 转为uint8
    else:
        # 如果已经是0-255范围，直接转换类型
        x_np = x_np.astype(np.uint8)
    
    # 5. 转换颜色通道（PyTorch默认RGB，OpenCV需要BGR）
    if x_np.shape[-1] == 3:
        x_np = cv2.cvtColor(x_np, cv2.COLOR_RGB2BGR)
    
    # 6. 显示图像
    cv2.imwrite(f"{name}.jpg", x_np)
    # cv2.imshow(f"{name}.jpg", x_np)
    # cv2.waitKey(wait_time)  # 等待按键或自动关闭
    # cv2.destroyAllWindows()  # 清理窗口
    return x_np  # 返回处理后的numpy数组（可选）


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * math.log10(255.0 / math.sqrt(mse))

def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    """calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    """
    if not img1.shape == img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError("Wrong input image dimensions.")
