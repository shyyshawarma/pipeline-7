# import os
# import json
# import warnings

# import torch
# import torch.nn.functional as F
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader

# from accelerate import Accelerator
# from tqdm import tqdm

# from config import Config
# from data import get_data
# from models import *
# from utils import *
# from torchsampler import ImbalancedDatasetSampler
# from metrics.uciqe import batch_uciqe

# from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
# from pytorch_msssim import ms_ssim
# from utils import VGGPerceptualLoss

# warnings.filterwarnings('ignore')

# # === Custom Losses ===

# class CharbonnierLoss(nn.Module):
#     def __init__(self, eps=1e-3):
#         super(CharbonnierLoss, self).__init__()
#         self.eps = eps

#     def forward(self, x, y):
#         return torch.mean(torch.sqrt((x - y) ** 2 + self.eps ** 2))


# class GradientLoss(nn.Module):
#     def __init__(self):
#         super(GradientLoss, self).__init__()
#         kernel = torch.FloatTensor([
#             [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
#             [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
#             [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
#         ]).unsqueeze(1)
#         self.register_buffer('weight', kernel)

#     def forward(self, x, y):
#         grad_x = F.conv2d(x, self.weight, padding=1, groups=3)
#         grad_y = F.conv2d(y, self.weight, padding=1, groups=3)
#         return F.l1_loss(grad_x, grad_y)


# class CompositeLoss(nn.Module):
#     def __init__(self):
#         super(CompositeLoss, self).__init__()
#         self.charb = CharbonnierLoss()
#         self.perceptual = VGGPerceptualLoss()
#         self.grad = GradientLoss()
#         self.ms_ssim = lambda x, y: 1 - ms_ssim(x, y, data_range=1.0, size_average=True)

#         self.weights = {
#             "Charbonnier": 0.2741,
#             "Perceptual": 0.1680,
#             "Gradient": 0.2222,
#             "MS-SSIM": 0.3357
#         }

#     def forward(self, pred, target):
#         l_charb = self.charb(pred, target)
#         l_perc = self.perceptual(pred, target)
#         l_grad = self.grad(pred, target)
#         l_ssim = self.ms_ssim(pred, target)

#         total = (
#             self.weights["Charbonnier"] * l_charb +
#             self.weights["Perceptual"] * l_perc +
#             self.weights["Gradient"] * l_grad +
#             self.weights["MS-SSIM"] * l_ssim
#         )
#         return total

# # === Training ===

# def train():
#     opt = Config('config.yml')
#     seed_everything(opt.OPTIM.SEED)

#     accelerator = Accelerator(log_with='wandb') if opt.OPTIM.WANDB else Accelerator()
#     if accelerator.is_local_main_process:
#         os.makedirs(opt.TRAINING.SAVE_DIR, exist_ok=True)
#     device = accelerator.device

#     config = {"dataset": opt.TRAINING.TRAIN_DIR}
#     accelerator.init_trackers("UW", config=config)

#     train_dataset = get_data(opt.TRAINING.TRAIN_DIR, opt.MODEL.INPUT, opt.MODEL.TARGET, 'train', opt.TRAINING.ORI,
#                              {'w': opt.TRAINING.PS_W, 'h': opt.TRAINING.PS_H})
#     trainloader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, num_workers=16,
#                              drop_last=False, pin_memory=True, sampler=ImbalancedDatasetSampler(train_dataset))

#     val_dataset = get_data(opt.TRAINING.VAL_DIR, opt.MODEL.INPUT, opt.MODEL.TARGET, 'test', opt.TRAINING.ORI,
#                            {'w': opt.TRAINING.PS_W, 'h': opt.TRAINING.PS_H})
#     testloader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=8,
#                             drop_last=False, pin_memory=True)

#     model = UIR_PolyKernel()

#     loss_fn = CompositeLoss().to(device)
#     optimizer_b = optim.AdamW(model.parameters(), lr=opt.OPTIM.LR_INITIAL, betas=(0.9, 0.999), eps=1e-8)
#     scheduler_b = optim.lr_scheduler.CosineAnnealingLR(optimizer_b, opt.OPTIM.NUM_EPOCHS, eta_min=opt.OPTIM.LR_MIN)

#     start_epoch = 1
#     best_psnr = 0
#     best_psnr_epoch = 1

#     if opt.TRAINING.RESUME and opt.TRAINING.WEIGHT is not None:
#         checkpoint = torch.load(opt.TRAINING.WEIGHT, map_location='cpu')
#         model.load_state_dict(checkpoint['state_dict'])

#         if 'optimizer' in checkpoint and 'epoch' in checkpoint:
#             optimizer_b.load_state_dict(checkpoint['optimizer'])
#             scheduler_b.load_state_dict(checkpoint['scheduler'])
#             start_epoch = checkpoint['epoch'] + 1
#             best_psnr = checkpoint.get('best_psnr', 0)
#             best_psnr_epoch = checkpoint.get('best_psnr_epoch', 1)

#         print(f"Resumed from epoch {start_epoch - 1}")

#     trainloader, testloader = accelerator.prepare(trainloader, testloader)
#     model, optimizer_b, scheduler_b = accelerator.prepare(model, optimizer_b, scheduler_b)

#     size = len(testloader)
#     early_stopping_patience = 15
#     epochs_no_improve = 0

#     for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
#         model.train()
#         for _, data in enumerate(tqdm(trainloader, disable=not accelerator.is_local_main_process)):
#             inp, tar = data[0].contiguous(), data[1]
#             optimizer_b.zero_grad()
#             res = model(inp)
#             loss = loss_fn(res, tar)
#             accelerator.backward(loss)
#             optimizer_b.step()

#         scheduler_b.step()

#         # === Print Loss Weights ===
#         with torch.no_grad():
#             print("\n🔍 Loss Weights after epoch {}:".format(epoch))
#             for name, val in loss_fn.weights.items():
#                 print(f"  {name}: {val:.4f}")

#         # === Validation ===
#         if epoch % opt.TRAINING.VAL_AFTER_EVERY == 0:
#             model.eval()
#             psnr = ssim = uciqe = ms_ssim_val = mse = 0

#             for _, data in enumerate(tqdm(testloader, disable=not accelerator.is_local_main_process)):
#                 inp, tar = data[0].contiguous(), data[1]
#                 with torch.no_grad():
#                     res = model(inp)

#                 res, tar = accelerator.gather((res, tar))
#                 psnr += peak_signal_noise_ratio(res, tar, data_range=1).item()
#                 ssim += structural_similarity_index_measure(res, tar, data_range=1).item()
#                 ms_ssim_val += ms_ssim(res, tar, data_range=1.0, size_average=True).item()
#                 mse += F.mse_loss(res, tar).item()
#                 uciqe += batch_uciqe(res)

#             psnr /= size
#             ssim /= size
#             ms_ssim_val /= size
#             mse /= size
#             uciqe /= size

#             accelerator.log({
#                 "PSNR": psnr,
#                 "SSIM": ssim,
#                 "MS-SSIM": ms_ssim_val,
#                 "MSE": mse,
#                 "UCIQE": uciqe
#             }, step=epoch)

#             if psnr > best_psnr:
#                 best_psnr = psnr
#                 best_psnr_epoch = epoch
#                 epochs_no_improve = 0

#                 save_checkpoint({
#                     'epoch': epoch,
#                     'state_dict': accelerator.unwrap_model(model).state_dict(),
#                     'optimizer': optimizer_b.state_dict(),
#                     'scheduler': scheduler_b.state_dict(),
#                     'best_psnr': best_psnr,
#                     'best_psnr_epoch': best_psnr_epoch
#                 }, epoch, opt.MODEL.SESSION, opt.TRAINING.SAVE_DIR)
#             else:
#                 epochs_no_improve += 1
#                 if epochs_no_improve >= early_stopping_patience:
#                     print(f"Early stopping triggered at epoch {epoch}")
#                     break

#             if accelerator.is_local_main_process:
#                 log_stats = (
#                     f"epoch: {epoch}, PSNR: {psnr:.4f}, SSIM: {ssim:.4f}, MS-SSIM: {ms_ssim_val:.4f}, "
#                     f"MSE: {mse:.6f}, UCIQE: {uciqe:.4f}, "
#                     f"best PSNR: {best_psnr:.4f}, best epoch: {best_psnr_epoch}"
#                 )
#                 print(log_stats)
#                 with open(os.path.join(opt.LOG.LOG_DIR, opt.TRAINING.LOG_FILE), 'a', encoding='utf-8') as f:
#                     f.write(json.dumps(log_stats) + '\n')

#     accelerator.end_training()


# if __name__ == '__main__':
#     train()

import os
import json
import warnings

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from accelerate import Accelerator
from tqdm import tqdm

from config import Config
from data import get_data
from models import *
from utils import *
from torchsampler import ImbalancedDatasetSampler
from metrics.uciqe import batch_uciqe

from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from pytorch_msssim import ms_ssim
from utils import VGGPerceptualLoss

warnings.filterwarnings('ignore')

# === Custom Losses ===

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        return torch.mean(torch.sqrt((x - y) ** 2 + self.eps ** 2))


class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()
        kernel = torch.FloatTensor([
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
        ]).unsqueeze(1)
        self.register_buffer('weight', kernel)

    def forward(self, x, y):
        grad_x = F.conv2d(x, self.weight, padding=1, groups=3)
        grad_y = F.conv2d(y, self.weight, padding=1, groups=3)
        return F.l1_loss(grad_x, grad_y)


class UWColorConstantLoss(nn.Module):
    def __init__(self):
        super(UWColorConstantLoss, self).__init__()

    def forward(self, image):
        # image: (B, C, H, W)
        means = image.mean(dim=[2, 3])  # (B, C)
        loss = 0.0
        for i in range(3):
            for j in range(i + 1, 3):
                loss += ((means[:, i] - means[:, j]) ** 2).mean()
        return loss


class CompositeLoss(nn.Module):
    def __init__(self):
        super(CompositeLoss, self).__init__()
        self.charb = CharbonnierLoss()
        self.perceptual = VGGPerceptualLoss()
        self.grad = GradientLoss()
        self.ms_ssim = lambda x, y: 1 - ms_ssim(x, y, data_range=1.0, size_average=True)
        self.color_const = UWColorConstantLoss()

        self.weights = {
            "Charbonnier": 0.2741,
            "Perceptual": 0.1680,
            "Gradient": 0.2222,
            "MS-SSIM": 0.3357,
            "ColorConstancy": 0.1500
        }

    def forward(self, pred, target):
        l_charb = self.charb(pred, target)
        l_perc = self.perceptual(pred, target)
        l_grad = self.grad(pred, target)
        l_ssim = self.ms_ssim(pred, target)
        l_color = self.color_const(pred)

        total = (
            self.weights["Charbonnier"] * l_charb +
            self.weights["Perceptual"] * l_perc +
            self.weights["Gradient"] * l_grad +
            self.weights["MS-SSIM"] * l_ssim +
            self.weights["ColorConstancy"] * l_color
        )
        return total


# === Training ===

def train():
    opt = Config('config.yml')
    seed_everything(opt.OPTIM.SEED)

    accelerator = Accelerator(log_with='wandb') if opt.OPTIM.WANDB else Accelerator()
    if accelerator.is_local_main_process:
        os.makedirs(opt.TRAINING.SAVE_DIR, exist_ok=True)
    device = accelerator.device

    config = {"dataset": opt.TRAINING.TRAIN_DIR}
    accelerator.init_trackers("UW", config=config)

    train_dataset = get_data(opt.TRAINING.TRAIN_DIR, opt.MODEL.INPUT, opt.MODEL.TARGET, 'train', opt.TRAINING.ORI,
                             {'w': opt.TRAINING.PS_W, 'h': opt.TRAINING.PS_H})
    trainloader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, num_workers=16,
                             drop_last=False, pin_memory=True, sampler=ImbalancedDatasetSampler(train_dataset))

    val_dataset = get_data(opt.TRAINING.VAL_DIR, opt.MODEL.INPUT, opt.MODEL.TARGET, 'test', opt.TRAINING.ORI,
                           {'w': opt.TRAINING.PS_W, 'h': opt.TRAINING.PS_H})
    testloader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=8,
                            drop_last=False, pin_memory=True)

    model = UIR_PolyKernel()

    loss_fn = CompositeLoss().to(device)
    optimizer_b = optim.AdamW(model.parameters(), lr=opt.OPTIM.LR_INITIAL, betas=(0.9, 0.999), eps=1e-8)
    scheduler_b = optim.lr_scheduler.CosineAnnealingLR(optimizer_b, opt.OPTIM.NUM_EPOCHS, eta_min=opt.OPTIM.LR_MIN)

    start_epoch = 1
    best_psnr = 0
    best_psnr_epoch = 1

    if opt.TRAINING.RESUME and opt.TRAINING.WEIGHT is not None:
        checkpoint = torch.load(opt.TRAINING.WEIGHT, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])

        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer_b.load_state_dict(checkpoint['optimizer'])
            scheduler_b.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint['epoch'] + 1
            best_psnr = checkpoint.get('best_psnr', 0)
            best_psnr_epoch = checkpoint.get('best_psnr_epoch', 1)

        print(f"Resumed from epoch {start_epoch - 1}")

    trainloader, testloader = accelerator.prepare(trainloader, testloader)
    model, optimizer_b, scheduler_b = accelerator.prepare(model, optimizer_b, scheduler_b)

    size = len(testloader)
    early_stopping_patience = 50
    epochs_no_improve = 0

    for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
        model.train()
        for _, data in enumerate(tqdm(trainloader, disable=not accelerator.is_local_main_process)):
            inp, tar = data[0].contiguous(), data[1]
            optimizer_b.zero_grad()
            res = model(inp)
            loss = loss_fn(res, tar)
            accelerator.backward(loss)
            optimizer_b.step()

        scheduler_b.step()

        # === Print Loss Weights ===
        with torch.no_grad():
            print("\n🔍 Loss Weights after epoch {}:".format(epoch))
            for name, val in loss_fn.weights.items():
                print(f"  {name}: {val:.4f}")

        # === Validation ===
        if epoch % opt.TRAINING.VAL_AFTER_EVERY == 0:
            model.eval()
            psnr = ssim = uciqe = ms_ssim_val = mse = 0

            for _, data in enumerate(tqdm(testloader, disable=not accelerator.is_local_main_process)):
                inp, tar = data[0].contiguous(), data[1]
                with torch.no_grad():
                    res = model(inp)

                res, tar = accelerator.gather((res, tar))
                psnr += peak_signal_noise_ratio(res, tar, data_range=1).item()
                ssim += structural_similarity_index_measure(res, tar, data_range=1).item()
                ms_ssim_val += ms_ssim(res, tar, data_range=1.0, size_average=True).item()
                mse += F.mse_loss(res, tar).item()
                uciqe += batch_uciqe(res)

            psnr /= size
            ssim /= size
            ms_ssim_val /= size
            mse /= size
            uciqe /= size

            accelerator.log({
                "PSNR": psnr,
                "SSIM": ssim,
                "MS-SSIM": ms_ssim_val,
                "MSE": mse,
                "UCIQE": uciqe
            }, step=epoch)

            if psnr > best_psnr:
                best_psnr = psnr
                best_psnr_epoch = epoch
                epochs_no_improve = 0

                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': accelerator.unwrap_model(model).state_dict(),
                    'optimizer': optimizer_b.state_dict(),
                    'scheduler': scheduler_b.state_dict(),
                    'best_psnr': best_psnr,
                    'best_psnr_epoch': best_psnr_epoch
                }, epoch, opt.MODEL.SESSION, opt.TRAINING.SAVE_DIR)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stopping_patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break

            if accelerator.is_local_main_process:
                log_stats = (
                    f"epoch: {epoch}, PSNR: {psnr:.4f}, SSIM: {ssim:.4f}, MS-SSIM: {ms_ssim_val:.4f}, "
                    f"MSE: {mse:.6f}, UCIQE: {uciqe:.4f}, "
                    f"best PSNR: {best_psnr:.4f}, best epoch: {best_psnr_epoch}"
                )
                print(log_stats)
                with open(os.path.join(opt.LOG.LOG_DIR, opt.TRAINING.LOG_FILE), 'a', encoding='utf-8') as f:
                    f.write(json.dumps(log_stats) + '\n')

    accelerator.end_training()


if __name__ == '__main__':
    train()

# import os
# import json
# import warnings

# import torch
# import torch.nn.functional as F
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader

# from accelerate import Accelerator
# from tqdm import tqdm

# from config import Config
# from data import get_data
# from models import *
# from utils import *
# from torchsampler import ImbalancedDatasetSampler
# from metrics.uciqe import batch_uciqe

# from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
# from pytorch_msssim import ms_ssim
# from utils import VGGPerceptualLoss

# warnings.filterwarnings('ignore')

# # === Custom Losses ===

# class CharbonnierLoss(nn.Module):
#     def __init__(self, eps=1e-3):
#         super(CharbonnierLoss, self).__init__()
#         self.eps = eps

#     def forward(self, x, y):
#         return torch.mean(torch.sqrt((x - y) ** 2 + self.eps ** 2))


# class GradientLoss(nn.Module):
#     def __init__(self):
#         super(GradientLoss, self).__init__()
#         kernel = torch.FloatTensor([
#             [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
#             [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
#             [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
#         ]).unsqueeze(1)
#         self.register_buffer('weight', kernel)

#     def forward(self, x, y):
#         grad_x = F.conv2d(x, self.weight, padding=1, groups=3)
#         grad_y = F.conv2d(y, self.weight, padding=1, groups=3)
#         return F.l1_loss(grad_x, grad_y)


# class UWColorConstantLoss(nn.Module):
#     def __init__(self):
#         super(UWColorConstantLoss, self).__init__()

#     def forward(self, image):
#         # image: (B, C, H, W)
#         means = image.mean(dim=[2, 3])  # (B, C)
#         loss = 0.0
#         for i in range(3):
#             for j in range(i + 1, 3):
#                 loss += ((means[:, i] - means[:, j]) ** 2).mean()
#         return loss


# class UWColorRelativeDispersionLoss(nn.Module):
#     def __init__(self):
#         super(UWColorRelativeDispersionLoss, self).__init__()

#     def forward(self, image):
#         # image: (B, C, H, W)
#         stds = image.std(dim=[2, 3])  # (B, C)
#         loss = 0.0
#         for i in range(3):
#             for j in range(i + 1, 3):
#                 loss += ((stds[:, i] - stds[:, j]) ** 2).mean()
#         return loss


# class CompositeLoss(nn.Module):
#     def __init__(self):
#         super(CompositeLoss, self).__init__()
#         self.charb = CharbonnierLoss()
#         self.perceptual = VGGPerceptualLoss()
#         self.grad = GradientLoss()
#         self.ms_ssim = lambda x, y: 1 - ms_ssim(x, y, data_range=1.0, size_average=True)
#         self.color_const = UWColorConstantLoss()
#         self.color_dispersion = UWColorRelativeDispersionLoss()  # ✅ New Loss

#         self.weights = {
#             "Charbonnier": 0.2741,
#             "Perceptual": 0.1680,
#             "Gradient": 0.2222,
#             "MS-SSIM": 0.3357,
#             "ColorConstancy": 0.1500,
#             "ColorDispersion": 0.1500  # ✅ New Weight
#         }

#     def forward(self, pred, target):
#         l_charb = self.charb(pred, target)
#         l_perc = self.perceptual(pred, target)
#         l_grad = self.grad(pred, target)
#         l_ssim = self.ms_ssim(pred, target)
#         l_color_const = self.color_const(pred)
#         l_color_disp = self.color_dispersion(pred)

#         total = (
#             self.weights["Charbonnier"] * l_charb +
#             self.weights["Perceptual"] * l_perc +
#             self.weights["Gradient"] * l_grad +
#             self.weights["MS-SSIM"] * l_ssim +
#             self.weights["ColorConstancy"] * l_color_const +
#             self.weights["ColorDispersion"] * l_color_disp
#         )
#         return total


# # === Training ===

# def train():
#     opt = Config('config.yml')
#     seed_everything(opt.OPTIM.SEED)

#     accelerator = Accelerator(log_with='wandb') if opt.OPTIM.WANDB else Accelerator()
#     if accelerator.is_local_main_process:
#         os.makedirs(opt.TRAINING.SAVE_DIR, exist_ok=True)
#     device = accelerator.device

#     config = {"dataset": opt.TRAINING.TRAIN_DIR}
#     accelerator.init_trackers("UW", config=config)

#     train_dataset = get_data(opt.TRAINING.TRAIN_DIR, opt.MODEL.INPUT, opt.MODEL.TARGET, 'train', opt.TRAINING.ORI,
#                              {'w': opt.TRAINING.PS_W, 'h': opt.TRAINING.PS_H})
#     trainloader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, num_workers=16,
#                              drop_last=False, pin_memory=True, sampler=ImbalancedDatasetSampler(train_dataset))

#     val_dataset = get_data(opt.TRAINING.VAL_DIR, opt.MODEL.INPUT, opt.MODEL.TARGET, 'test', opt.TRAINING.ORI,
#                            {'w': opt.TRAINING.PS_W, 'h': opt.TRAINING.PS_H})
#     testloader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=8,
#                             drop_last=False, pin_memory=True)

#     model = UIR_PolyKernel()

#     loss_fn = CompositeLoss().to(device)
#     optimizer_b = optim.AdamW(model.parameters(), lr=opt.OPTIM.LR_INITIAL, betas=(0.9, 0.999), eps=1e-8)
#     scheduler_b = optim.lr_scheduler.CosineAnnealingLR(optimizer_b, opt.OPTIM.NUM_EPOCHS, eta_min=opt.OPTIM.LR_MIN)

#     start_epoch = 1
#     best_psnr = 0
#     best_psnr_epoch = 1

#     if opt.TRAINING.RESUME and opt.TRAINING.WEIGHT is not None:
#         checkpoint = torch.load(opt.TRAINING.WEIGHT, map_location='cpu')
#         model.load_state_dict(checkpoint['state_dict'])

#         if 'optimizer' in checkpoint and 'epoch' in checkpoint:
#             optimizer_b.load_state_dict(checkpoint['optimizer'])
#             scheduler_b.load_state_dict(checkpoint['scheduler'])
#             start_epoch = checkpoint['epoch'] + 1
#             best_psnr = checkpoint.get('best_psnr', 0)
#             best_psnr_epoch = checkpoint.get('best_psnr_epoch', 1)

#         print(f"Resumed from epoch {start_epoch - 1}")

#     trainloader, testloader = accelerator.prepare(trainloader, testloader)
#     model, optimizer_b, scheduler_b = accelerator.prepare(model, optimizer_b, scheduler_b)

#     size = len(testloader)
#     early_stopping_patience = 25
#     epochs_no_improve = 0

#     for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
#         model.train()
#         for _, data in enumerate(tqdm(trainloader, disable=not accelerator.is_local_main_process)):
#             inp, tar = data[0].contiguous(), data[1]
#             optimizer_b.zero_grad()
#             res = model(inp)
#             loss = loss_fn(res, tar)
#             accelerator.backward(loss)
#             optimizer_b.step()

#         scheduler_b.step()

#         # === Print Loss Weights ===
#         with torch.no_grad():
#             print("\n🔍 Loss Weights after epoch {}:".format(epoch))
#             for name, val in loss_fn.weights.items():
#                 print(f"  {name}: {val:.4f}")

#         # === Validation ===
#         if epoch % opt.TRAINING.VAL_AFTER_EVERY == 0:
#             model.eval()
#             psnr = ssim = uciqe = ms_ssim_val = mse = 0

#             for _, data in enumerate(tqdm(testloader, disable=not accelerator.is_local_main_process)):
#                 inp, tar = data[0].contiguous(), data[1]
#                 with torch.no_grad():
#                     res = model(inp)

#                 res, tar = accelerator.gather((res, tar))
#                 psnr += peak_signal_noise_ratio(res, tar, data_range=1).item()
#                 ssim += structural_similarity_index_measure(res, tar, data_range=1).item()
#                 ms_ssim_val += ms_ssim(res, tar, data_range=1.0, size_average=True).item()
#                 mse += F.mse_loss(res, tar).item()
#                 uciqe += batch_uciqe(res)

#             psnr /= size
#             ssim /= size
#             ms_ssim_val /= size
#             mse /= size
#             uciqe /= size

#             accelerator.log({
#                 "PSNR": psnr,
#                 "SSIM": ssim,
#                 "MS-SSIM": ms_ssim_val,
#                 "MSE": mse,
#                 "UCIQE": uciqe
#             }, step=epoch)

#             if psnr > best_psnr:
#                 best_psnr = psnr
#                 best_psnr_epoch = epoch
#                 epochs_no_improve = 0

#                 save_checkpoint({
#                     'epoch': epoch,
#                     'state_dict': accelerator.unwrap_model(model).state_dict(),
#                     'optimizer': optimizer_b.state_dict(),
#                     'scheduler': scheduler_b.state_dict(),
#                     'best_psnr': best_psnr,
#                     'best_psnr_epoch': best_psnr_epoch
#                 }, epoch, opt.MODEL.SESSION, opt.TRAINING.SAVE_DIR)
#             else:
#                 epochs_no_improve += 1
#                 if epochs_no_improve >= early_stopping_patience:
#                     print(f"Early stopping triggered at epoch {epoch}")
#                     break

#             if accelerator.is_local_main_process:
#                 log_stats = (
#                     f"epoch: {epoch}, PSNR: {psnr:.4f}, SSIM: {ssim:.4f}, MS-SSIM: {ms_ssim_val:.4f}, "
#                     f"MSE: {mse:.6f}, UCIQE: {uciqe:.4f}, "
#                     f"best PSNR: {best_psnr:.4f}, best epoch: {best_psnr_epoch}"
#                 )
#                 print(log_stats)
#                 with open(os.path.join(opt.LOG.LOG_DIR, opt.TRAINING.LOG_FILE), 'a', encoding='utf-8') as f:
#                     f.write(json.dumps(log_stats) + '\n')

#     accelerator.end_training()


# if __name__ == '__main__':
#     train()
