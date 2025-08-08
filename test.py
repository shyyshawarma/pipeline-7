import json
import warnings
import os
import torch
import torchvision
import numpy as np
import lpips
import cv2
import math
from piq import gmsd as piq_gmsd
from niqe_utils import calculate_niqe
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchmetrics.functional import (
    peak_signal_noise_ratio, structural_similarity_index_measure,
    mean_squared_error, multiscale_structural_similarity_index_measure
)
from metrics.uciqe import batch_uciqe
from tqdm import tqdm
from config import Config
from data import get_data
from models import *
from utils import *

warnings.filterwarnings('ignore')


def tensor_to_image(tensor):
    image = tensor.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return image


def delta_e(img1, img2):
    from skimage.color import rgb2lab, deltaE_cie76
    lab1 = rgb2lab(img1)
    lab2 = rgb2lab(img2)
    return np.mean(deltaE_cie76(lab1, lab2))


def mad(img1, img2):
    return np.mean(np.abs(img1.astype(np.float32) - img2.astype(np.float32)))


def euclidean_distance(img1, img2):
    return np.linalg.norm(img1.astype(np.float32) - img2.astype(np.float32))


def vif(img1, img2):
    from sewar.full_ref import vifp
    return vifp(img1, img2)


from piq import fsim as piq_fsim

def fsim(img1, img2):
    img1_tensor = torch.tensor(img1.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
    img2_tensor = torch.tensor(img2.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
    img1_tensor, img2_tensor = img1_tensor.to(device), img2_tensor.to(device)
    return piq_fsim(img1_tensor, img2_tensor, data_range=1.).item()


def gmsd(img1, img2):
    img1_tensor = torch.tensor(img1.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
    img2_tensor = torch.tensor(img2.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
    img1_tensor, img2_tensor = img1_tensor.to(device), img2_tensor.to(device)
    return piq_gmsd(img1_tensor, img2_tensor, data_range=1.).item()


def cie2000(img1, img2):
    from skimage.color import rgb2lab, deltaE_ciede2000
    lab1 = rgb2lab(img1)
    lab2 = rgb2lab(img2)
    return np.mean(deltaE_ciede2000(lab1, lab2))


def entropy_score(img):
    from skimage.measure import shannon_entropy
    return shannon_entropy(img)


def compute_no_ref_metrics(img):
    import piq
    from piq import brisque

    img_tensor = torch.tensor(img.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
    img_tensor = img_tensor.to(device)

    return {
        'NIQE': calculate_niqe(img, crop_border=0, input_order='HWC', convert_to='y'),
        'BRISQUE': brisque(img_tensor).item(),
        'Entropy': entropy_score(img)
    }


# Refactored UCIQE/UIQM functions from Code 2
from skimage import color
from skimage.util import view_as_windows
from scipy.signal import convolve2d

def uicm(rgb_image):
    img_double = rgb_image.astype(np.float64)
    R, G, B = img_double[:, :, 0], img_double[:, :, 1], img_double[:, :, 2]
    RG = R - G
    YB = (R + G) / 2 - B

    rg_flat = RG.flatten()
    yb_flat = YB.flatten()

    alpha_l, alpha_r = 0.1, 0.1
    k = len(rg_flat)
    start_idx = int(np.floor(alpha_l * k))
    end_idx = int(np.ceil(k * (1 - alpha_r)))

    rg_trimmed = np.sort(rg_flat)[start_idx:end_idx]
    yb_trimmed = np.sort(yb_flat)[start_idx:end_idx]

    mean_rg, std_rg = np.mean(rg_trimmed), np.std(rg_trimmed)
    mean_yb, std_yb = np.mean(yb_trimmed), np.std(yb_trimmed)

    uicm_val = -0.0268 * np.sqrt(mean_rg ** 2 + mean_yb ** 2) + 0.1586 * np.sqrt(std_rg ** 2 + std_yb ** 2)
    return uicm_val


def uism(rgb_image):
    img_double = rgb_image.astype(np.float64)
    R, G, B = img_double[:, :, 0], img_double[:, :, 1], img_double[:, :, 2]

    hx = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float64)
    hy = hx.T

    def sobel_energy(channel):
        sx = convolve2d(channel, hx, 'same', 'symm')
        sy = convolve2d(channel, hy, 'same', 'symm')
        return np.abs(sx + sy)

    patch_size = 5
    def eme(channel):
        windows = view_as_windows(channel, (patch_size, patch_size), step=patch_size)
        max_vals = np.max(windows, axis=(2, 3))
        min_vals = np.min(windows, axis=(2, 3))
        ratio = np.ones_like(max_vals)
        valid = (min_vals > 0) & (max_vals > 0)
        ratio[valid] = max_vals[valid] / min_vals[valid]
        return np.mean(np.log(ratio)) * 2

    return 0.299 * eme(sobel_energy(R)) + 0.587 * eme(sobel_energy(G)) + 0.114 * eme(sobel_energy(B))


def uiconm(rgb_image):
    img_double = rgb_image.astype(np.float64)
    R, G, B = img_double[:, :, 0], img_double[:, :, 1], img_double[:, :, 2]

    patch_size = 5
    def amee(channel):
        windows = view_as_windows(channel, (patch_size, patch_size), step=patch_size)
        max_vals = np.max(windows, axis=(2, 3))
        min_vals = np.min(windows, axis=(2, 3))
        numerator = max_vals - min_vals
        denominator = max_vals + min_vals
        valid = denominator > 0
        x = np.zeros_like(numerator)
        x[valid] = numerator[valid] / denominator[valid]

        log_term = np.zeros_like(x)
        log_term[x > 0] = np.log(x[x > 0])

        return np.mean(x * log_term)

    return np.abs(amee(R)) + np.abs(amee(G)) + np.abs(amee(B))


def calculate_uiqm(rgb_image):
    c1, c2, c3 = 0.0282, 0.2953, 3.5753
    return c1 * uicm(rgb_image) + c2 * uism(rgb_image) + c3 * uiconm(rgb_image)


def test():
    global device
    opt = Config('config.yml')
    seed_everything(opt.OPTIM.SEED)

    accelerator = Accelerator()
    device = accelerator.device

    val_dir = opt.TESTING.VAL_DIR
    val_dataset = get_data(val_dir, opt.TESTING.INPUT, opt.TESTING.TARGET, 'test', opt.TRAINING.ORI,
                           {'w': opt.TRAINING.PS_W, 'h': opt.TRAINING.PS_H})
    testloader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False,
                            pin_memory=True)

    model = UIR_PolyKernel()
    load_checkpoint(model, opt.TESTING.WEIGHT)
    model, testloader = accelerator.prepare(model, testloader)
    model.eval()
    lpips_model = lpips.LPIPS(net='alex').to(device)
    metrics = {
        'PSNR': 0, 'SSIM': 0, 'MSE': 0, 'RMSE': 0, 'Delta-E': 0, 'MAD': 0,
        'GMSD': 0, 'FSIM': 0, 'VIF': 0, 'CIEDE2000': 0, 'MS-SSIM': 0, 'LPIPS': 0,
        'NIQE': 0, 'BRISQUE': 0, 'Entropy': 0,
        'UCIQE': 0, 'UIQM': 0, 'UICM': 0
    }

    for _, test_data in enumerate(tqdm(testloader)):
        inp = test_data[0].to(device)
        tar = test_data[1].to(device)

        with torch.no_grad():
            res = model(inp)

        if not os.path.isdir(opt.TESTING.RESULT_DIR):
            os.makedirs(opt.TESTING.RESULT_DIR)
        torchvision.utils.save_image(res, os.path.join(opt.TESTING.RESULT_DIR, test_data[2][0]))

        res_img = tensor_to_image(res)
        tar_img = tensor_to_image(tar)

        # Tensor-based metrics
        metrics['PSNR'] += peak_signal_noise_ratio(res, tar, data_range=1).item()
        metrics['SSIM'] += structural_similarity_index_measure(res, tar, data_range=1).item()
        mse_val = mean_squared_error(res, tar).item()
        metrics['MSE'] += mse_val
        
        metrics['MS-SSIM'] += multiscale_structural_similarity_index_measure(res, tar, data_range=1).item()
        metrics['LPIPS'] += lpips_model(res, tar).mean().item()

        # NumPy-based metrics
        metrics['Delta-E'] += delta_e(res_img, tar_img)
        metrics['MAD'] += mad(res_img, tar_img)
        metrics['GMSD'] += gmsd(res_img, tar_img)
        metrics['FSIM'] += fsim(res_img, tar_img)
        metrics['VIF'] += vif(res_img, tar_img)
        metrics['CIEDE2000'] += cie2000(res_img, tar_img)

        # No-ref
        no_ref = compute_no_ref_metrics(res_img)
        for key in ['NIQE', 'BRISQUE', 'Entropy']:
            metrics[key] += no_ref[key]

        # UIQ/UCIQE
        # uciqe_score = calculate_uciqe(res_img)
        uiqm_score = calculate_uiqm(res_img)
        uicm_score = uicm(res_img)
        metrics['UCIQE'] += batch_uciqe(res)
        metrics['UIQM'] += uiqm_score
        metrics['UICM'] += uicm_score
    
    n = len(testloader)
    for k in metrics:
        metrics[k] /= n
    metrics['RMSE'] = math.sqrt(metrics['MSE'])
    
    test_info = f"Test Result on {opt.MODEL.SESSION}, checkpoint {opt.TESTING.WEIGHT}, testing data {opt.TESTING.VAL_DIR}"
    print(test_info)
    for k, v in metrics.items():
        print(f"{k}: {float(v):.4f}")

    # Ensure all metric values are converted to native Python floats
    metrics = {k: float(v) for k, v in metrics.items()}

    with open(os.path.join(opt.LOG.LOG_DIR, opt.TESTING.LOG_FILE), mode='a', encoding='utf-8') as f:
        f.write(json.dumps(test_info) + '\n')
        f.write(json.dumps(metrics) + '\n')

if __name__ == '__main__':
    os.makedirs('result', exist_ok=True)
    test()