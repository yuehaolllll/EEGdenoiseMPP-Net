import torch
import torch.nn as nn
import torch.nn.functional as F


class ZeroPhaseConv1d(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(in_c, out_c, kernel_size, padding=padding, dilation=dilation)

    def forward(self, x): return self.conv(x)


class MPR_ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        mid_c = out_c // 4
        self.b1 = ZeroPhaseConv1d(in_c, mid_c, kernel_size=7, dilation=8)
        self.b2 = ZeroPhaseConv1d(in_c, mid_c, kernel_size=5, dilation=4)
        self.b3 = ZeroPhaseConv1d(in_c, mid_c, kernel_size=3, dilation=1)
        self.b4 = nn.Conv1d(in_c, out_c - 3 * mid_c, kernel_size=1)
        self.bn = nn.BatchNorm1d(out_c)
        self.prelu = nn.PReLU()
        self.res_link = nn.Conv1d(in_c, out_c, kernel_size=1)
        self.dropout = nn.Dropout1d(p=0.3)

    def forward(self, x):
        res = self.res_link(x)
        out = torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)
        return self.dropout(self.prelu(self.bn(out + res)))


class ZeroPhasePool(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv1d(in_c, out_c, 4, stride=2, padding=1)

    def forward(self, x): return self.conv(x)


class MPR_Net_ZeroPhase(nn.Module):
    def __init__(self, base_c=24):
        super().__init__()
        self.enc1 = MPR_ResidualBlock(1, base_c)
        self.pool1 = ZeroPhasePool(base_c, base_c)
        self.enc2 = MPR_ResidualBlock(base_c, base_c * 2)
        self.pool2 = ZeroPhasePool(base_c * 2, base_c * 2)
        self.enc3 = MPR_ResidualBlock(base_c * 2, base_c * 4)
        self.pool3 = ZeroPhasePool(base_c * 4, base_c * 4)
        self.bottleneck = MPR_ResidualBlock(base_c * 4, base_c * 8)
        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2), ZeroPhaseConv1d(base_c * 8, base_c * 4, 3))
        self.dec3 = MPR_ResidualBlock(base_c * 8, base_c * 4)
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2), ZeroPhaseConv1d(base_c * 4, base_c * 2, 3))
        self.dec2 = MPR_ResidualBlock(base_c * 4, base_c * 2)
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2), ZeroPhaseConv1d(base_c * 2, base_c, 3))
        self.final = nn.Sequential(ZeroPhaseConv1d(base_c * 2, base_c, 3), nn.PReLU(), nn.Conv1d(base_c, 1, 1))

    def forward(self, x):
        # 针对 512 长度适配：512 是 8 的倍数，直接下采样 3 次 (8倍) 不需要额外 Padding
        # 如果你坚持要 reflect padding 增强边缘，可以使用 (4,4) 保持 8 的倍数
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        bn = self.bottleneck(self.pool3(e3))
        d3 = self.dec3(torch.cat([self.up3(bn), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        out = self.final(torch.cat([self.up1(d2), e1], dim=1))
        return out


# 你的自定义 Loss
class EEGDenoiseLoss(nn.Module):
    def __init__(self, window_size=512):  # 改为 512
        super(EEGDenoiseLoss, self).__init__()
        self.register_buffer('window', torch.hann_window(window_size))

    def forward(self, pred, target):
        loss_time = F.smooth_l1_loss(pred, target)
        pred_fft = torch.fft.rfft(pred * self.window, dim=-1, norm='ortho')
        target_fft = torch.fft.rfft(target * self.window, dim=-1, norm='ortho')
        loss_freq = F.l1_loss(torch.log1p(torch.abs(pred_fft)), torch.log1p(torch.abs(target_fft)))

        # 相关性损失
        p_c = pred - pred.mean(dim=-1, keepdim=True)
        t_c = target - target.mean(dim=-1, keepdim=True)
        pearson = (p_c * t_c).sum(dim=-1) / (
                    torch.sqrt((p_c ** 2).sum(dim=-1)) * torch.sqrt((t_c ** 2).sum(dim=-1)) + 1e-8)
        loss_corr = torch.mean(1.0 - pearson)

        return 0.8 * loss_time + 1.5 * loss_freq + 1.2 * loss_corr


# ==========================================
# 变体 1: 单尺度消融版 (Single-Scale Variant)
# 目的：证明并行多尺度分支的必要性
# ==========================================
class SingleScaleBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        # 只保留一个标准卷积分支，不使用扩张卷积和多尺度
        self.conv = nn.Conv1d(in_c, out_c, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(out_c)
        self.prelu = nn.PReLU()
        self.res_link = nn.Conv1d(in_c, out_c, kernel_size=1)
        self.dropout = nn.Dropout1d(p=0.3)

    def forward(self, x):
        res = self.res_link(x)
        out = self.conv(x)
        return self.dropout(self.prelu(self.bn(out + res)))

class MPR_Net_SingleScale(nn.Module):
    def __init__(self, base_c=24):
        super().__init__()
        # 使用 SingleScaleBlock 替换原来的 MPR_ResidualBlock
        self.enc1 = SingleScaleBlock(1, base_c)
        self.pool1 = ZeroPhasePool(base_c, base_c)
        self.enc2 = SingleScaleBlock(base_c, base_c * 2)
        self.pool2 = ZeroPhasePool(base_c * 2, base_c * 2)
        self.enc3 = SingleScaleBlock(base_c * 2, base_c * 4)
        self.pool3 = ZeroPhasePool(base_c * 4, base_c * 4)
        self.bottleneck = SingleScaleBlock(base_c * 4, base_c * 8)
        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2), ZeroPhaseConv1d(base_c * 8, base_c * 4, 3))
        self.dec3 = SingleScaleBlock(base_c * 8, base_c * 4)
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2), ZeroPhaseConv1d(base_c * 4, base_c * 2, 3))
        self.dec2 = SingleScaleBlock(base_c * 4, base_c * 2)
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2), ZeroPhaseConv1d(base_c * 2, base_c, 3))
        self.final = nn.Sequential(ZeroPhaseConv1d(base_c * 2, base_c, 3), nn.PReLU(), nn.Conv1d(base_c, 1, 1))

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        bn = self.bottleneck(self.pool3(e3))
        d3 = self.dec3(torch.cat([self.up3(bn), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        out = self.final(torch.cat([self.up1(d2), e1], dim=1))
        return out


# ==========================================
# 变体 2: 无残差消融版 (No-Residual Variant)
# 目的：证明残差连接对深层信号恢复的价值
# ==========================================
class NoResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        mid_c = out_c // 4
        self.b1 = ZeroPhaseConv1d(in_c, mid_c, kernel_size=7, dilation=8)
        self.b2 = ZeroPhaseConv1d(in_c, mid_c, kernel_size=5, dilation=4)
        self.b3 = ZeroPhaseConv1d(in_c, mid_c, kernel_size=3, dilation=1)
        self.b4 = nn.Conv1d(in_c, out_c - 3 * mid_c, kernel_size=1)
        self.bn = nn.BatchNorm1d(out_c)
        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout1d(p=0.3)

    def forward(self, x):
        # 去掉 res_link 和加法，直接拼接后输出
        out = torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)
        return self.dropout(self.prelu(self.bn(out)))

class MP_Net(nn.Module):
    def __init__(self, base_c=24):
        super().__init__()
        # 使用 NoResidualBlock 替换原来的 MPR_ResidualBlock
        self.enc1 = NoResidualBlock(1, base_c)
        self.pool1 = ZeroPhasePool(base_c, base_c)
        self.enc2 = NoResidualBlock(base_c, base_c * 2)
        self.pool2 = ZeroPhasePool(base_c * 2, base_c * 2)
        self.enc3 = NoResidualBlock(base_c * 2, base_c * 4)
        self.pool3 = ZeroPhasePool(base_c * 4, base_c * 4)
        self.bottleneck = NoResidualBlock(base_c * 4, base_c * 8)
        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2), ZeroPhaseConv1d(base_c * 8, base_c * 4, 3))
        self.dec3 = NoResidualBlock(base_c * 8, base_c * 4)
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2), ZeroPhaseConv1d(base_c * 4, base_c * 2, 3))
        self.dec2 = NoResidualBlock(base_c * 4, base_c * 2)
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2), ZeroPhaseConv1d(base_c * 2, base_c, 3))
        self.final = nn.Sequential(ZeroPhaseConv1d(base_c * 2, base_c, 3), nn.PReLU(), nn.Conv1d(base_c, 1, 1))

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        bn = self.bottleneck(self.pool3(e3))
        d3 = self.dec3(torch.cat([self.up3(bn), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        out = self.final(torch.cat([self.up1(d2), e1], dim=1))
        return out