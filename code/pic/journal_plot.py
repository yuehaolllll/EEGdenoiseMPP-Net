import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.signal import welch

# ==================== 1. 路径修复逻辑 ====================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from benchmark_networks.Network_structure import Simple_CNN, Complex_CNN
    from Novel_CNN.Novel_CNN import Novel_CNN
    from MP_Net.MP_Net import MPR_Net_ZeroPhase, EEGDenoiseLoss, MPR_Net_SingleScale, MP_Net

    print("✅ 模型模块加载成功")
except ImportError as e:
    print(f"❌ 导入失败: {e}")

# ==================== 2. 全任务配置 ====================
# 配置要绘图的 [Mode, Task] 组合
PLOT_CONFIGS = [
    ['Official', 'EOG'],
    ['Official', 'EMG'],
    ['SelfMethod', 'Robust_Mixed']
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_dir = os.path.join(project_root, 'pic')
os.makedirs(save_dir, exist_ok=True)


def calculate_rrmse(p, t):
    return torch.sqrt(torch.mean((p - t) ** 2)) / torch.sqrt(torch.mean(t ** 2))


# ==================== 3. 核心循环 ====================
for mode, task in PLOT_CONFIGS:
    print(f"\n>>>> 正在处理任务: {mode} - {task} <<<<")

    # --- A. 加载对应数据 ---
    if mode == 'SelfMethod':
        data_folder = 'processed_data_SelfMethod'
    else:
        data_folder = f'processed_data_{task}'

    data_path = os.path.join(project_root, 'benchmark_networks', data_folder)
    tx = torch.from_numpy(np.load(os.path.join(data_path, 'test_x.npy'))).float().unsqueeze(1)
    ty = torch.from_numpy(np.load(os.path.join(data_path, 'test_y.npy'))).float().unsqueeze(1)

    total_samples = tx.shape[0]
    samples_per_snr = total_samples // 10
    SNR_LEVELS = np.linspace(-7.0, 2.0, num=10)

    # --- B. 收集模型结果 ---
    results = {}
    models_to_test = ['Simple_CNN', 'Novel_CNN', 'MP_Net']

    for net_name in models_to_test:
        if net_name == 'MP_Net':
            model = MP_Net(base_c=24).to(DEVICE)
        elif net_name == 'Simple_CNN':
            model = Simple_CNN(512).to(DEVICE)
        elif net_name == 'Novel_CNN':
            model = Novel_CNN(512).to(DEVICE)

        weight_path = os.path.join(project_root, 'benchmark_networks', 'results', mode, task, net_name,
                                   'denoise_model.pth')

        if not os.path.exists(weight_path):
            print(f"   ⚠️ 跳过 {net_name}: 权重缺失")
            continue

        model.load_state_dict(torch.load(weight_path, map_location=DEVICE), strict=False)
        model.eval()

        snr_rrmse = []
        with torch.no_grad():
            for i in range(10):
                start, end = i * samples_per_snr, (i + 1) * samples_per_snr
                batch_x, batch_y = tx[start:end].to(DEVICE), ty[start:end].to(DEVICE)
                pred = model(batch_x)
                snr_rrmse.append(calculate_rrmse(pred, batch_y).item())

        results[net_name] = snr_rrmse

    # --- C. 绘制折线图 ---
    plt.figure(figsize=(8, 6))
    colors = ['#34495e', '#e67e22', '#e74c3c']
    markers = ['o', 's', 'D']
    for i, (name, values) in enumerate(results.items()):
        label = "MPP-Net (Ours)" if name == 'MP_Net' else name
        plt.plot(SNR_LEVELS, values, label=label, marker=markers[i], color=colors[i], markersize=7, linewidth=1.5)

    plt.xlabel('Input SNR (dB)')
    plt.ylabel('Temporal RRMSE')
    plt.title(f'RRMSE Comparison: {task} ({mode})', fontweight='bold')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(save_dir, f'snr_line_{mode}_{task}.png'), dpi=300)
    plt.close()

    # --- D. 绘制波形图 (选取第一个 SNR 级别的样本) ---
    mpr_weight = os.path.join(project_root, 'benchmark_networks', 'results', mode, task, 'MP_Net', 'denoise_model.pth')
    model = MP_Net(base_c=24).to(DEVICE)
    model.load_state_dict(torch.load(mpr_weight, map_location=DEVICE), strict=False)
    model.eval()

    idx = 37 # 选取一个固定样本
    dirty_val = tx[idx:idx + 1].to(DEVICE)
    clean_val = ty[idx:idx + 1].to(DEVICE)
    with torch.no_grad():
        pred_val = model(dirty_val).cpu().numpy().flatten()

    raw = dirty_val.cpu().numpy().flatten()
    gt = clean_val.cpu().numpy().flatten()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(raw, color='#bdc3c7', alpha=0.7, label='Noisy EEG')
    ax1.plot(gt, color='#2c3e50', lw=1.8, label='Ground Truth (Clean)')
    ax1.plot(pred_val, color='#e67e22', lw=1.5, linestyle='--', label='MPP-Net (Ours)')
    ax1.set_title(f'Temporal Waveform Alignment: {task}', fontweight='bold')
    ax1.set_xlabel('Time Samples')
    ax1.set_ylabel('Amplitude (μV)')
    ax1.legend(loc='upper right')

    f, px = welch(raw, fs=256, nperseg=256)
    _, py = welch(gt, fs=256, nperseg=256)
    _, pp = welch(pred_val, fs=256, nperseg=256)
    ax2.semilogy(f, px, color='#bdc3c7', alpha=0.7)
    ax2.semilogy(f, py, color='#2c3e50', lw=1.8)
    ax2.semilogy(f, pp, color='#e67e22', lw=1.5, linestyle='--')
    ax2.set_xlim(0, 60)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Power Spectral Density (μV²/Hz)')
    ax2.set_title('Frequency Domain Preservation (PSD)', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'waveform_psd_{mode}_{task}.png'), dpi=300)
    plt.close()

print(f"\n🎉 所有图表已生成，请查看文件夹: {save_dir}")