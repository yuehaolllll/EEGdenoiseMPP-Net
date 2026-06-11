"""
论文美化图: Fig 3 (SNR), Fig 4 (Waveform+PSD), Fig 5 (Efficiency)
"""
import torch, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, os, sys
from scipy.signal import welch

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from MP_Net.MP_Net import MP_Net
PIC_DIR = os.path.join(project_root, 'pic')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
                      'font.family': 'sans-serif', 'figure.dpi': 200})


def calc_rrmse(p, t):
    return torch.sqrt(torch.mean((p-t)**2)) / torch.sqrt(torch.mean(t**2))


# ═══════════════════════════════════════════
# FIGURE 3: SNR Robustness — 1×3 combined
# ═══════════════════════════════════════════
def make_fig3():
    PLOT_CONFIGS = [
        ['Official', 'EOG',       '(a) EOG Task'],
        ['Official', 'EMG',       '(b) EMG Task'],
        ['SelfMethod', 'Robust_Mixed', '(c) Robust Mixed Task'],
    ]
    SNR_LEVELS = np.linspace(-7.0, 2.0, num=10)
    colors = {'Simple_CNN': '#7f8c8d', 'Novel_CNN': '#e67e22', 'MP_Net': '#e74c3c'}
    labels = {'Simple_CNN': 'Simple CNN', 'Novel_CNN': 'Novel CNN', 'MP_Net': 'MPP-Net (Ours)'}
    markers = {'Simple_CNN': 's', 'Novel_CNN': '^', 'MP_Net': 'o'}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)

    for ax_idx, (mode, task, title) in enumerate(PLOT_CONFIGS):
        ax = axes[ax_idx]
        data_folder = 'processed_data_SelfMethod' if mode == 'SelfMethod' else f'processed_data_{task}'
        data_path = os.path.join(project_root, 'benchmark_networks', data_folder)
        tx = torch.from_numpy(np.load(os.path.join(data_path, 'test_x.npy'))).float().unsqueeze(1)
        ty = torch.from_numpy(np.load(os.path.join(data_path, 'test_y.npy'))).float().unsqueeze(1)
        total_samples = tx.shape[0]
        samples_per_snr = total_samples // 10

        for net_name in ['Simple_CNN', 'Novel_CNN', 'MP_Net']:
            if net_name == 'MP_Net':
                model = MP_Net(base_c=24).to(DEVICE)
            elif net_name == 'Simple_CNN':
                from benchmark_networks.Network_structure import Simple_CNN
                model = Simple_CNN(512).to(DEVICE)
            else:
                from Novel_CNN.Novel_CNN import Novel_CNN
                model = Novel_CNN(512).to(DEVICE)

            wpath = os.path.join(project_root, 'benchmark_networks', 'results',
                                 mode, task, net_name, 'denoise_model.pth')
            if not os.path.exists(wpath): continue
            model.load_state_dict(torch.load(wpath, map_location=DEVICE), strict=False)
            model.eval()

            vals = []
            with torch.no_grad():
                for i in range(10):
                    s, e = i * samples_per_snr, (i + 1) * samples_per_snr
                    bx, by = tx[s:e].to(DEVICE), ty[s:e].to(DEVICE)
                    vals.append(calc_rrmse(model(bx), by).item())

            ax.plot(SNR_LEVELS, vals, label=labels[net_name], color=colors[net_name],
                    marker=markers[net_name], markersize=6, linewidth=1.5, markevery=2)

        ax.set_xlabel('Input SNR (dB)')
        if ax_idx == 0: ax.set_ylabel('Temporal RRMSE')
        ax.set_title(title, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    handles, labels_list = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_list, loc='upper center', ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, 1.02), framealpha=0.9)
    fig.suptitle('RRMSE vs. Input SNR Across Tasks', fontweight='bold', fontsize=14, y=1.08)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(PIC_DIR, 'fig3_snr_combined.png'), dpi=350, bbox_inches='tight')
    plt.close()
    print('[Fig 3] Saved: fig3_snr_combined.png')


# ═══════════════════════════════════════════
# FIGURE 4: Waveform + PSD — 3×2 combined
# ═══════════════════════════════════════════
def make_fig4():
    configs = [
        ['Official', 'EOG',       '(a) EOG Task'],
        ['Official', 'EMG',       '(b) EMG Task'],
        ['SelfMethod', 'Robust_Mixed', '(c) Robust Mixed Task'],
    ]

    fig, axes = plt.subplots(3, 2, figsize=(14, 13))

    for row, (mode, task, title) in enumerate(configs):
        data_folder = 'processed_data_SelfMethod' if mode == 'SelfMethod' else f'processed_data_{task}'
        data_path = os.path.join(project_root, 'benchmark_networks', data_folder)
        tx = torch.from_numpy(np.load(os.path.join(data_path, 'test_x.npy'))).float().unsqueeze(1)
        ty = torch.from_numpy(np.load(os.path.join(data_path, 'test_y.npy'))).float().unsqueeze(1)

        wpath = os.path.join(project_root, 'benchmark_networks', 'results',
                             mode, task, 'MP_Net', 'denoise_model.pth')
        model = MP_Net(base_c=24).to(DEVICE)
        model.load_state_dict(torch.load(wpath, map_location=DEVICE), strict=False)
        model.eval()

        idx = 37
        dirty = tx[idx:idx+1].to(DEVICE); clean = ty[idx:idx+1].to(DEVICE)
        with torch.no_grad(): pred = model(dirty).cpu().numpy().flatten()
        raw = dirty.cpu().numpy().flatten(); gt = clean.cpu().numpy().flatten()
        t = np.arange(len(raw)) / 256

        # Waveform
        ax = axes[row, 0]
        ax.plot(t, raw, color='#bdc3c7', lw=0.6, alpha=0.7, label='Noisy')
        ax.plot(t, gt, color='#2c3e50', lw=1.3, label='Clean')
        ax.plot(t, pred, color='#e67e22', lw=1.2, linestyle='--', label='MPP-Net')
        ax.set_xlabel('Time (s)'); ax.set_ylabel('Amplitude')
        ax.set_title(f'{title} — Waveform', fontweight='bold')
        ax.legend(fontsize=7, loc='upper right'); ax.grid(True, alpha=0.25)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

        # PSD
        ax = axes[row, 1]
        f_n, p_n = welch(raw, fs=256, nperseg=256)
        f_c, p_c = welch(gt, fs=256, nperseg=256)
        f_p, p_p = welch(pred, fs=256, nperseg=256)
        ax.semilogy(f_n, p_n, color='#bdc3c7', lw=0.6, alpha=0.7, label='Noisy')
        ax.semilogy(f_c, p_c, color='#2c3e50', lw=1.3, label='Clean')
        ax.semilogy(f_p, p_p, color='#e67e22', lw=1.2, linestyle='--', label='MPP-Net')
        ax.set_xlim(0, 60); ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('PSD')
        ax.set_title(f'{title} — PSD', fontweight='bold')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.25)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    fig.suptitle('Temporal Waveform Alignment and Spectral Fidelity', fontweight='bold', fontsize=14, y=1.01)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(os.path.join(PIC_DIR, 'fig4_waveform_psd_combined.png'), dpi=350, bbox_inches='tight')
    plt.close()
    print('[Fig 4] Saved: fig4_waveform_psd_combined.png')


# ═══════════════════════════════════════════
# FIGURE 5: Efficiency Scatter — beautified
# ═══════════════════════════════════════════
def make_fig5():
    models = ['fcNN', 'RNN-LSTM', 'Simple CNN', 'Complex CNN', 'Novel CNN', 'MPP-Net']
    params = [1.05, 0.79, 16.82, 8.46, 33.56, 0.32]
    eog_rrmse = [0.5415, 0.5685, 0.3625, 0.3574, 0.4188, 0.2939]
    colors = ['#95a5a6','#95a5a6','#3498db','#3498db','#e67e22','#e74c3c']
    sizes  = [80, 80, 120, 120, 150, 280]
    markers = ['o','o','s','s','^','*']

    fig, ax = plt.subplots(figsize=(10, 7))

    for i in range(len(models)):
        ax.scatter(params[i], eog_rrmse[i], s=sizes[i], c=colors[i], marker=markers[i],
                   edgecolors='black' if i < 5 else '#c0392b', linewidths=0.8 if i < 5 else 2.0,
                   zorder=5 if i < 5 else 10, alpha=0.85)
        offset = (12, -18) if i == 5 else (10, 8)
        fw = 'bold' if i == 5 else 'normal'
        ax.annotate(models[i], (params[i], eog_rrmse[i]), xytext=offset, textcoords='offset points',
                    fontsize=9, fontweight=fw, color='#c0392b' if i == 5 else '#2c3e50')

    ax.set_xscale('log')
    ax.set_xlabel('Parameters (Million) — Log Scale', fontweight='bold', fontsize=12)
    ax.set_ylabel('Temporal RRMSE (EOG Task)', fontweight='bold', fontsize=12)
    ax.set_title('Denoising Performance vs. Model Complexity', fontweight='bold', fontsize=14)

    # Optimal zone shading
    ax.axhspan(0.26, 0.32, xmin=0.02, xmax=0.25, color='#27ae60', alpha=0.08, zorder=0)
    ax.annotate('Optimal Zone', xy=(0.4, 0.288), fontsize=9, color='#27ae60',
                fontweight='bold', fontstyle='italic')

    # Arrow pointing to MPP-Net
    ax.annotate('', xy=(0.32, 0.2939), xytext=(0.8, 0.33),
                arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.5))
    ax.annotate('Best efficiency:\n0.31M params,\nRRMSE=0.294', xy=(1.0, 0.335),
                fontsize=9, color='#c0392b', fontweight='bold', ha='left')

    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(os.path.join(PIC_DIR, 'fig5_efficiency.png'), dpi=350, bbox_inches='tight')
    plt.close()
    print('[Fig 5] Saved: fig5_efficiency.png')


if __name__ == '__main__':
    print(f'Device: {DEVICE}')
    make_fig3()
    make_fig4()
    make_fig5()
    print('\nAll done!')
