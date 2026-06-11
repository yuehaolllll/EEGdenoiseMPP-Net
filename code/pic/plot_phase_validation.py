"""
零相位验证：互相关峰值滞后分析
对比 MPP-Net (对称填充) vs Simple_CNN (非对称填充) 的相位保真度
"""
import torch, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, os, sys
from scipy.signal import correlate

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
from MP_Net.MP_Net import MP_Net
from benchmark_networks.Network_structure import Simple_CNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = os.path.join(current_dir, 'selfdata_results')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def compute_lag(denoised, clean):
    """计算互相关峰值滞后 (lag). lag=0 表示完美零相位对齐."""
    best_lag = 0; best_corr = -999
    for lag in range(-20, 21):
        d_shifted = np.roll(denoised, lag)
        cc = np.corrcoef(d_shifted, clean)[0, 1]
        if cc > best_corr:
            best_corr = cc; best_lag = lag
    return best_lag


def main():
    TASKS = [('EOG', 'Official'), ('EMG', 'Official'), ('Robust_Mixed', 'SelfMethod')]

    plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12, 'font.family': 'sans-serif', 'figure.dpi': 200})

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, (task, mode) in enumerate(TASKS):
        ax = axes[idx]

        # Load test data
        data_folder = 'processed_data_SelfMethod' if mode == 'SelfMethod' else f'processed_data_{task}'
        data_path = os.path.join(parent_dir, 'benchmark_networks', data_folder)
        tx = torch.from_numpy(np.load(os.path.join(data_path, 'test_x.npy'))).float().unsqueeze(1)
        ty = torch.from_numpy(np.load(os.path.join(data_path, 'test_y.npy'))).float().unsqueeze(1)

        # Load MPP-Net
        mpp_path = os.path.join(parent_dir, 'benchmark_networks', 'results',
                                 mode, task, 'MP_Net', 'denoise_model.pth')
        mpp = MP_Net(base_c=24).to(DEVICE)
        mpp.load_state_dict(torch.load(mpp_path, map_location=DEVICE), strict=False)
        mpp.eval()

        # Load Simple_CNN
        scnn_path = os.path.join(parent_dir, 'benchmark_networks', 'results',
                                  mode, task, 'Simple_CNN', 'denoise_model.pth')
        scnn = Simple_CNN(512).to(DEVICE)
        scnn.load_state_dict(torch.load(scnn_path, map_location=DEVICE), strict=False)
        scnn.eval()

        # Sample test segments (use first 200)
        n_samples = min(200, tx.shape[0])
        mpp_lags = []; scnn_lags = []
        with torch.no_grad():
            for i in range(n_samples):
                inp = tx[i:i+1].to(DEVICE)
                clean = ty[i].numpy().flatten()
                mpp_out = mpp(inp).cpu().numpy().flatten()
                scnn_out = scnn(inp).cpu().numpy().flatten()
                mpp_lags.append(compute_lag(mpp_out, clean))
                scnn_lags.append(compute_lag(scnn_out, clean))

        mpp_lags = np.array(mpp_lags); scnn_lags = np.array(scnn_lags)

        # Plot histogram
        bins = np.arange(-20, 22, 1)
        ax.hist(mpp_lags, bins=bins, alpha=0.7, color='#e74c3c', label=f'MPP-Net (lag={np.mean(np.abs(mpp_lags)):.2f})')
        ax.hist(scnn_lags, bins=bins, alpha=0.6, color='#7f8c8d', label=f'Simple CNN (lag={np.mean(np.abs(scnn_lags)):.2f})')
        ax.axvline(0, color='black', linestyle='--', lw=1, alpha=0.5)
        ax.set_xlabel('Cross-Correlation Lag (samples)')
        ax.set_ylabel('Count')
        ax.set_title(f'{task}', fontweight='bold')
        ax.legend(fontsize=9)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    fig.suptitle('Phase Fidelity Validation: Cross-Correlation Peak Lag Distribution\n(Lag=0 = Perfect Zero-Phase Alignment)',
                 fontweight='bold', fontsize=13, y=1.03)
    plt.tight_layout()
    sp = os.path.join(OUTPUT_DIR, 'phase_validation_lag.png')
    plt.savefig(sp, dpi=350, bbox_inches='tight')
    plt.close()

    # Summary stats
    print('=== Phase Validation Summary ===')
    for task, mode in TASKS:
        mpp_path = os.path.join(parent_dir, 'benchmark_networks', 'results', mode, task, 'MP_Net', 'denoise_model.pth')
        scnn_path = os.path.join(parent_dir, 'benchmark_networks', 'results', mode, task, 'Simple_CNN', 'denoise_model.pth')
        mpp = MP_Net(base_c=24).to(DEVICE); mpp.load_state_dict(torch.load(mpp_path, map_location=DEVICE), strict=False); mpp.eval()
        scnn = Simple_CNN(512).to(DEVICE); scnn.load_state_dict(torch.load(scnn_path, map_location=DEVICE), strict=False); scnn.eval()
        data_folder = 'processed_data_SelfMethod' if mode == 'SelfMethod' else f'processed_data_{task}'
        data_path = os.path.join(parent_dir, 'benchmark_networks', data_folder)
        tx = torch.from_numpy(np.load(os.path.join(data_path, 'test_x.npy'))).float().unsqueeze(1)
        ty = torch.from_numpy(np.load(os.path.join(data_path, 'test_y.npy'))).float().unsqueeze(1)
        n = min(200, tx.shape[0])
        ml, sl = [], []
        with torch.no_grad():
            for i in range(n):
                inp = tx[i:i+1].to(DEVICE); clean = ty[i].numpy().flatten()
                ml.append(compute_lag(mpp(inp).cpu().numpy().flatten(), clean))
                sl.append(compute_lag(scnn(inp).cpu().numpy().flatten(), clean))
        ml, sl = np.array(ml), np.array(sl)
        mpp_zero = np.sum(ml == 0) / len(ml) * 100
        scnn_zero = np.sum(sl == 0) / len(sl) * 100
        print(f'{task}: MPP-Net lag@0={mpp_zero:.1f}%, mean|lag|={np.mean(np.abs(ml)):.2f} | Simple_CNN lag@0={scnn_zero:.1f}%, mean|lag|={np.mean(np.abs(sl)):.2f}')

    print(f'\nSaved: {sp}')


if __name__ == '__main__':
    main()
