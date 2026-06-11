"""
去噪效果组合展示图 — 4种伪迹 × 时域+频域，2×4网格
"""
import torch, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, os, sys, re
from scipy.signal import welch, butter, filtfilt, resample

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from MP_Net.MP_Net import MP_Net

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SELFDATA_DIR = os.path.join(parent_dir, 'selfdata')
OUTPUT_DIR = os.path.join(current_dir, 'selfdata_results')
SELF_FS, TARGET_FS, WIN, HOP, DUR = 250, 256, 512, 256, 8
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_ch1(filepath):
    data = np.genfromtxt(filepath, delimiter=',', skip_header=3,
                         comments='#', filling_values=np.nan)
    sig = data[:, 1]; sig = sig[~np.isnan(sig)]
    sig = resample(sig, int(len(sig) * TARGET_FS / SELF_FS))
    nyq = TARGET_FS / 2
    b, a = butter(4, [0.5 / nyq, 80 / nyq], btype='band')
    return filtfilt(b, a, sig)


def denoise(model, signal):
    n = len(signal); pad = WIN // 2
    p = np.pad(signal, (pad, pad + HOP), mode='reflect')
    out, wgt = np.zeros_like(p), np.zeros_like(p)
    hw = np.hanning(WIN)
    model.eval()
    for s in range(0, len(p) - WIN + 1, HOP):
        seg = p[s:s + WIN]
        seg_n = (seg - seg.mean()) / (seg.std() + 1e-8)
        x = torch.from_numpy(seg_n).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
        with torch.no_grad(): d = model(x).cpu().numpy().flatten()
        d = d * seg.std() + seg.mean()
        out[s:s+WIN] += d * hw; wgt[s:s+WIN] += hw
    return (out / (wgt + 1e-8))[pad:pad+n]


def main():
    # Load models
    model_paths = {
        'EMG': os.path.join(parent_dir, 'benchmark_networks', 'results',
                            'Official', 'EMG', 'MP_Net', 'denoise_model.pth'),
        'EOG': os.path.join(parent_dir, 'benchmark_networks', 'results',
                            'Official', 'EOG', 'MP_Net', 'denoise_model.pth'),
        'Robust_Mixed': os.path.join(parent_dir, 'benchmark_networks', 'results',
                                     'SelfMethod', 'Robust_Mixed', 'MP_Net', 'denoise_model.pth'),
    }
    M = {}
    for k, p in model_paths.items():
        m = MP_Net(base_c=24).to(DEVICE)
        m.load_state_dict(torch.load(p, map_location=DEVICE), strict=False)
        m.eval(); M[k] = m

    # Config: (data_type, subdir, best_model, file_index)
    configs = [
        ('EMG',    'EMG',    'Robust_Mixed', 0),
        ('EOG',    'EOG',    'Robust_Mixed', 0),
        ('Mix',    'Mix',    'Robust_Mixed', 0),
        ('Motion', 'Motion', 'Robust_Mixed', 0),
    ]

    plt.rcParams.update({
        'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
        'font.family': 'sans-serif', 'figure.dpi': 200,
    })

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))

    for idx, (dtype, dname, best_mk, _) in enumerate(configs):
        d = os.path.join(SELFDATA_DIR, dname)
        fs = sorted([f for f in os.listdir(d) if f.endswith('.csv')],
                    key=lambda x: int(re.search(r'(\d+)', x).group()))
        fn = fs[0]  # First sample
        seg = load_ch1(os.path.join(d, fn))
        n = DUR * TARGET_FS
        seg = seg[len(seg)//2 - n//2 : len(seg)//2 + n//2]
        out = denoise(M[best_mk], seg)
        t = np.arange(len(seg)) / TARGET_FS

        # ---- Row 1: Time domain ----
        ax = axes[0, idx]
        ax.plot(t, seg, color='#e74c3c', lw=0.5, alpha=0.7, label='Raw')
        ax.plot(t, out, color='#2980b9', lw=1.0, label=f'{best_mk}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (μV)')
        ax.set_title(f'{dtype} Artifacts — Time Domain', fontweight='bold')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(alpha=0.25)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # ---- Row 2: Frequency domain ----
        ax = axes[1, idx]
        f_r, p_r = welch(seg, fs=TARGET_FS, nperseg=512)
        f_d, p_d = welch(out, fs=TARGET_FS, nperseg=512)
        ax.semilogy(f_r, p_r, color='#e74c3c', lw=1.0, alpha=0.7, label='Raw')
        ax.semilogy(f_d, p_d, color='#2980b9', lw=1.2, label=f'{best_mk}')
        ax.set_xlim(0, 60)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('PSD')
        ax.set_title(f'{dtype} Artifacts — Frequency Domain', fontweight='bold')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle('MPP-Net Denoising on Self-Collected Real Artifacts (Representative Samples)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    sp = os.path.join(OUTPUT_DIR, 'denoising_gallery.png')
    plt.savefig(sp, dpi=350, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {sp}")


if __name__ == '__main__':
    main()
