"""
自采集真实噪声数据 — 三模型全交叉去噪分析
EMG / EOG / Robust_Mixed 模型 × EMG / EOG / Mix / Motion 数据
"""
import torch, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, os, sys, re
from scipy.signal import welch, butter, filtfilt, resample

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
from MP_Net.MP_Net import MP_Net

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SELFDATA_DIR = os.path.join(parent_dir, 'selfdata')
OUTPUT_DIR   = os.path.join(parent_dir, 'pic', 'selfdata_results')
SELF_FS, TARGET_FS, WIN, HOP, DUR = 250, 256, 512, 256, 10
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


def seg_mid(signal):
    n = DUR * TARGET_FS
    if len(signal) > n:
        s = (len(signal) - n) // 2
        return signal[s:s + n]
    return signal[:n]


def metrics(raw, denoised):
    rr = np.sqrt(np.mean(raw**2)); ro = np.sqrt(np.mean(denoised**2))
    r_red = (rr - ro) / rr * 100
    f_r, p_r = welch(raw, fs=TARGET_FS, nperseg=512)
    f_d, p_d = welch(denoised, fs=TARGET_FS, nperseg=512)
    m = f_r <= 80
    e_raw = np.trapezoid(p_r[m], f_r[m]); e_out = np.trapezoid(p_d[m], f_d[m])
    e_red = (e_raw - e_out) / e_raw * 100 if e_raw > 0 else 0
    return rr, ro, r_red, e_raw, e_out, e_red


def plot_one(raw, denoised, title, spath):
    """单张: 时域+频域对比."""
    t = np.arange(len(raw)) / TARGET_FS
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(t, raw, color='#e74c3c', lw=0.5, alpha=0.8, label='Raw')
    ax1.plot(t, denoised, color='#2980b9', lw=1.2, label='MPP-Net')
    ax1.set_xlabel('Time (s)'); ax1.set_ylabel('Amplitude (μV)')
    ax1.set_title(f'Time — {title}'); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
    fr, pr = welch(raw, fs=TARGET_FS, nperseg=512)
    fd, pd = welch(denoised, fs=TARGET_FS, nperseg=512)
    ax2.semilogy(fr, pr, color='#e74c3c', lw=1.2, label='Raw')
    ax2.semilogy(fd, pd, color='#2980b9', lw=1.2, label='MPP-Net')
    ax2.set_xlim(0, 80); ax2.set_xlabel('Frequency (Hz)'); ax2.set_ylabel('PSD')
    ax2.set_title(f'Freq — {title}'); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(spath, dpi=300); plt.close()


def plot_cross(raw, outputs, title, spath):
    """raw vs 3 model outputs overlaid."""
    t = np.arange(len(raw)) / TARGET_FS
    colors = {'EMG': '#e74c3c', 'EOG': '#27ae60', 'Robust_Mixed': '#2980b9'}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    ax1.plot(t, raw, color='gray', lw=0.5, alpha=0.6, label='Raw')
    for mk, out in outputs.items():
        ax1.plot(t, out, color=colors[mk], lw=0.8, label=mk)
    ax1.set_xlabel('Time (s)'); ax1.set_ylabel('Amplitude')
    ax1.set_title(f'Time — {title}'); ax1.legend(fontsize=7); ax1.grid(alpha=0.3)

    fr, pr = welch(raw, fs=TARGET_FS, nperseg=512)
    ax2.semilogy(fr, pr, color='gray', lw=0.8, alpha=0.6, label='Raw')
    for mk, out in outputs.items():
        fd, pd = welch(out, fs=TARGET_FS, nperseg=512)
        ax2.semilogy(fd, pd, color=colors[mk], lw=0.8, label=mk)
    ax2.set_xlim(0, 80); ax2.set_xlabel('Frequency (Hz)'); ax2.set_ylabel('PSD')
    ax2.set_title(f'Freq — {title}'); ax2.legend(fontsize=7); ax2.grid(alpha=0.3)

    plt.tight_layout(); plt.savefig(spath, dpi=300); plt.close()


def main():
    print(f"Device: {DEVICE}")
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
        m.eval(); M[k] = m; print(f"  {k} loaded")

    data_dirs = {'EMG': 'EMG', 'EOG': 'EOG', 'Mix': 'Mix', 'Motion': 'Motion'}
    all_model_names = ['EMG', 'EOG', 'Robust_Mixed']
    results = {}  # {data_type: {model: {fname: {metrics}}}}

    for dtype, dname in data_dirs.items():
        d = os.path.join(SELFDATA_DIR, dname)
        fs = sorted([f for f in os.listdir(d) if f.endswith('.csv')],
                    key=lambda x: int(re.search(r'(\d+)', x).group()))

        results[dtype] = {mk: {} for mk in all_model_names}

        for mk in all_model_names:
            print(f"\n[{dtype}] {len(fs)} files ← {mk}")
            for fn in fs:
                fp = os.path.join(d, fn)
                seg = seg_mid(load_ch1(fp))
                out = denoise(M[mk], seg)
                rr, ro, rred, er, eo, ered = metrics(seg, out)
                results[dtype][mk][fn] = {
                    'rr':rr, 'ro':ro, 'rred':rred, 'er':er, 'eo':eo, 'ered':ered
                }
                print(f"  {fn}: RMS {rr:.1f}→{ro:.1f} ({rred:+.1f}%)  E {er:.0f}→{eo:.0f} ({ered:+.1f}%)")
                # 每个文件-模型组合单独出图
                outname = f'selfdata_{dtype}_{fn[:-4]}_{mk}.png'
                plot_one(seg, out, f'{dtype} {fn} ← {mk}',
                         os.path.join(OUTPUT_DIR, outname))
    # --- Per-data cross-model comparison plots ---
    print("\nGenerating cross-model overlay plots...")
    for dtype, dname in data_dirs.items():
        d = os.path.join(SELFDATA_DIR, dname)
        fs = sorted([f for f in os.listdir(d) if f.endswith('.csv')],
                    key=lambda x: int(re.search(r'(\d+)', x).group()))
        # 每类用第1个文件做三模型对比图
        fn = fs[0]
        fp = os.path.join(d, fn)
        seg = seg_mid(load_ch1(fp))
        outputs = {mk: denoise(M[mk], seg) for mk in all_model_names}
        plot_cross(seg, outputs, f'{dtype} {fn}',
                   os.path.join(OUTPUT_DIR, f'cross_{dtype}_{fn[:-4]}.png'))

    # --- Report ---
    print("\n\n" + "=" * 75)
    R = ["三模型全交叉对比报告 (EMG / EOG / Robust_Mixed)",
         "=" * 75,
         f"4类数据 × 3模型 = 12组, 每组5文件 | {DUR}s/segment",
         ""]

    # ---- Summary: per data type, which model is best ----
    R.append("═══ 按数据类型汇总: 各模型平均 RMS 衰减 ═══")
    R.append(f"{'数据类型':<10} {'EMG模型':>14} {'EOG模型':>14} {'Robust_Mixed':>14} {'最优模型':<14}")
    R.append("-" * 75)

    overall_wins = {mk: 0 for mk in all_model_names}
    for dtype in ['EMG', 'EOG', 'Mix', 'Motion']:
        row = f"  {dtype:<8}"
        best_avg = -999; best_mk = ''
        for mk in all_model_names:
            vals = [v['rred'] for v in results[dtype][mk].values()]
            avg = np.mean(vals); std = np.std(vals)
            row += f"  {avg:.1f}%±{std:.1f}%"
            if avg > best_avg: best_avg = avg; best_mk = mk
        row += f"  → {best_mk}"
        overall_wins[best_mk] += 1
        R.append(row)

    # ---- Overall winner ----
    R.append(f"\n  各类型最优模型统计: {overall_wins}")
    sorted_models = sorted(overall_wins.items(), key=lambda x: x[1], reverse=True)

    # ---- Detailed per-file table ----
    R.append("")
    R.append("═══ 全部详细数据 (每文件 × 3模型) ═══")
    R.append(f"{'文件':<30} {'模型':<14} {'RMS_raw':>8} {'RMS_out':>8} {'ΔRMS':>7}  {'E_raw':>10} {'E_out':>10} {'ΔE':>7}")
    R.append("-" * 105)
    for dtype in ['EMG', 'EOG', 'Mix', 'Motion']:
        for mk in all_model_names:
            for fn, v in results[dtype][mk].items():
                R.append(
                    f"{dtype}/{fn:<24} {mk:<14} {v['rr']:>8.1f} {v['ro']:>8.1f} "
                    f"{v['rred']:>+6.1f}%  {v['er']:>10.0f} {v['eo']:>10.0f} {v['ered']:>+6.1f}%"
                )

    # ---- Cross-type generalization ----
    R.append("")
    R.append("═══ 跨类型泛化分析 ═══")
    R.append(f"{'模型':<14} {'EMG数据':>12} {'EOG数据':>12} {'Mix数据':>12} {'Motion数据':>12} {'平均':>12}")
    R.append("-" * 70)
    for mk in all_model_names:
        row = f"  {mk:<12}"
        all_avgs = []
        for dtype in ['EMG', 'EOG', 'Mix', 'Motion']:
            vals = [v['rred'] for v in results[dtype][mk].values()]
            avg = np.mean(vals)
            all_avgs.append(avg)
            row += f"  {avg:>+8.1f}%"
        row += f"  {np.mean(all_avgs):>+8.1f}%"
        R.append(row)

    R.extend([
        "",
        "═" * 75,
        "结论:",
        "  1. 各数据类型上, RMS衰减最高的模型即为该类型的最优去噪模型",
        "  2. 匹配模型(Match): 例如EMG模型→EMG数据, 预期应表现最佳",
        "  3. 跨类型泛化(Cross): 模型在非训练类型上的表现反映其普适性",
        "  4. 总体最优: 在所有4种数据类型上胜出次数最多的模型",
    ])

    txt = '\n'.join(R)
    print("\n" + txt)
    with open(os.path.join(OUTPUT_DIR, 'analysis_report.txt'), 'w', encoding='utf-8') as f:
        f.write(txt)
    print(f"\nDone → {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
