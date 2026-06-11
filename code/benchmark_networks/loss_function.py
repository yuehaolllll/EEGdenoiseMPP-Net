import torch
import torch.nn as nn
import torch.nn.functional as F


def denoise_loss_mse(denoise, clean):
    return F.mse_loss(denoise, clean)


def denoise_loss_rmse(denoise, clean):
    return torch.sqrt(F.mse_loss(denoise, clean))


def denoise_loss_rrmset(denoise, clean):
    rmse1 = denoise_loss_rmse(denoise, clean)
    # clean 与 全 0 信号的 RMSE 实际上就是 clean 的 RMS
    rmse2 = torch.sqrt(torch.mean(clean**2))
    return rmse1 / (rmse2 + 1e-8)

def calculate_cc(pred, target):
    """计算相关系数 (Batch 维度平均)"""
    p_c = pred - torch.mean(pred, dim=-1, keepdim=True)
    t_c = target - torch.mean(target, dim=-1, keepdim=True)
    num = torch.sum(p_c * t_c, dim=-1)
    den = torch.sqrt(torch.sum(p_c ** 2, dim=-1) * torch.sum(t_c ** 2, dim=-1) + 1e-8)
    return torch.mean(num / den).item()

def calculate_spectral_rrmse(pred, target):
    """计算频域 RRMSE (使用功率谱密度 PSD)"""
    # 计算 FFT 的幅度谱
    pred_fft = torch.abs(torch.fft.rfft(pred, dim=-1))
    target_fft = torch.abs(torch.fft.rfft(target, dim=-1))

    # 频域 RRMSE 公式
    num = torch.sqrt(torch.mean((target_fft - pred_fft) ** 2))
    den = torch.sqrt(torch.mean(target_fft ** 2))
    return (num / den).item()

def calculate_snr(signal, noise_residue):
    """计算单个信号的 SNR"""
    p_sig = torch.mean(signal**2, dim=-1)
    p_noise = torch.mean(noise_residue**2, dim=-1)
    # 转换为 dB
    snr = 10 * torch.log10(p_sig / (p_noise + 1e-8))
    return torch.mean(snr).item()

def get_snr_gain(pred, target, noisy):
    """计算 SNR 提升量 (越高越好)"""
    # 输入 SNR: 干净信号 vs (含噪信号 - 干净信号)
    snr_in = calculate_snr(target, noisy - target)
    # 输出 SNR: 干净信号 vs (去噪信号 - 干净信号)
    snr_out = calculate_snr(target, pred - target)
    return snr_out - snr_in, snr_in, snr_out