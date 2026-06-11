import torch
import numpy as np
import os
import sys
from scipy.stats import pearsonr


# ==================== 1. 相位误差核心函数 ====================
def calculate_phase_error(pred, target, fs=256, freq_range=(0, 60)):
    """
    计算去噪信号与干净信号之间的平均绝对相位误差 (MAPE-Phase)
    参数:
        pred: 模型预测信号 (Batch, 1, 512) 或 (Batch, 512)
        target: 原始干净信号 (Batch, 1, 512) 或 (Batch, 512)
        fs: 采样率 (Hz)
        freq_range: 关注的频率范围 (Hz)，脑电通常为 0-60Hz
    """
    # 转为 numpy 数组并去掉通道维度
    p = pred.detach().cpu().numpy().squeeze()
    t = target.detach().cpu().numpy().squeeze()

    # 1. 快速傅里叶变换
    fft_p = np.fft.rfft(p, axis=-1)
    fft_t = np.fft.rfft(t, axis=-1)

    # 2. 提取相位角 (弧度)
    phase_p = np.angle(fft_p)
    phase_t = np.angle(fft_t)

    # 3. 计算环形相位差
    # 使用 atan2(sin(d), cos(d)) 确保相位差在 [-pi, pi] 之间，避免边界跳变
    phase_diff = phase_p - phase_t
    dist = np.arctan2(np.sin(phase_diff), np.cos(phase_diff))

    # 4. 频率掩码：仅计算 0-60Hz 范围内的误差
    freqs = np.fft.rfftfreq(p.shape[-1], d=1 / fs)
    mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])

    # 5. 取选定频段内所有样本的绝对值平均
    abs_error = np.abs(dist[..., mask])
    return np.mean(abs_error)


def calculate_cc(p, t):
    """计算相关系数"""
    p_flat = p.detach().cpu().numpy().flatten()
    t_flat = t.detach().cpu().numpy().flatten()
    return pearsonr(p_flat, t_flat)[0]


# ==================== 2. 集成到你的测试循环 ====================
# 这里模仿你提供的路径修复逻辑
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from benchmark_networks.Network_structure import Simple_CNN
from Novel_CNN.Novel_CNN import Novel_CNN
from MP_Net.MP_Net import MP_Net

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PLOT_CONFIGS = [['Official', 'EOG'], ['Official', 'EMG']]

# 存储最终结果用于展示
summary_table = []

for mode, task in PLOT_CONFIGS:
    print(f"\n📊 正在评估任务: {mode} - {task}")

    # 加载数据 (假设数据在相同位置)
    data_folder = f'processed_data_{task}'
    data_path = os.path.join(project_root, 'benchmark_networks', data_folder)
    tx = torch.from_numpy(np.load(os.path.join(data_path, 'test_x.npy'))).float().unsqueeze(1).to(DEVICE)
    ty = torch.from_numpy(np.load(os.path.join(data_path, 'test_y.npy'))).float().unsqueeze(1).to(DEVICE)

    models_to_test = ['Simple_CNN', 'Novel_CNN', 'MP_Net']

    for net_name in models_to_test:
        # 模型初始化
        if net_name == 'MP_Net':
            model = MP_Net(base_c=24).to(DEVICE)
        elif net_name == 'Simple_CNN':
            model = Simple_CNN(512).to(DEVICE)
        elif net_name == 'Novel_CNN':
            model = Novel_CNN(512).to(DEVICE)

        weight_path = os.path.join(project_root, 'benchmark_networks', 'results', mode, task, net_name,
                                   'denoise_model.pth')
        if not os.path.exists(weight_path): continue

        model.load_state_dict(torch.load(weight_path, map_location=DEVICE), strict=False)
        model.eval()

        with torch.no_grad():
            pred = model(tx)

            # 计算三大硬指标
            rrmse = (torch.sqrt(torch.mean((pred - ty) ** 2)) / torch.sqrt(torch.mean(ty ** 2))).item()
            cc = calculate_cc(pred, ty)
            phase_err = calculate_phase_error(pred, ty, fs=256)

            summary_table.append({
                'Task': task,
                'Model': 'MPP-Net (Ours)' if net_name == 'MP_Net' else net_name,
                'RRMSE': f"{rrmse:.4f}",
                'CC': f"{cc:.4f}",
                'Phase_Err(rad)': f"{phase_err:.4f}"
            })

# ==================== 3. 打印对比表格 ====================
print("\n" + "=" * 60)
print(f"{'Task':<8} | {'Model':<18} | {'RRMSE':<8} | {'CC':<8} | {'Phase_Err':<10}")
print("-" * 60)
for row in summary_table:
    print(f"{row['Task']:<8} | {row['Model']:<18} | {row['RRMSE']:<8} | {row['CC']:<8} | {row['Phase_Err(rad)']:<10}")
print("=" * 60)