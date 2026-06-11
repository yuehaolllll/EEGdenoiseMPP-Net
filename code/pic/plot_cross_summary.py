"""
顶刊级三模型全交叉对比汇总图
"""
import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, os
from matplotlib.patches import FancyBboxPatch

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'selfdata_results')
REPORT = os.path.join(OUTPUT_DIR, 'analysis_report.txt')


def parse_report(path):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    models = ['EMG', 'EOG', 'Robust_Mixed']
    results = {}
    for line in text.split('\n'):
        parts = line.strip().split()
        if len(parts) < 7: continue
        try:
            dtype = parts[0].split('/')[0]
            mk = parts[1]; rms = float(parts[4].replace('%','').replace('+',''))
            if mk not in models: continue
            results.setdefault(dtype, {}).setdefault(mk, []).append(rms)
        except (ValueError, IndexError): continue
    return results


def main():
    results = parse_report(REPORT)
    data_types = ['EMG', 'EOG', 'Mix', 'Motion']
    models = ['EMG', 'EOG', 'Robust_Mixed']

    plt.rcParams.update({
        'font.size': 13, 'axes.titlesize': 15, 'axes.labelsize': 14,
        'legend.fontsize': 12, 'xtick.labelsize': 12, 'ytick.labelsize': 12,
        'font.family': 'sans-serif', 'figure.dpi': 200,
    })

    colors = {'EMG': '#E85D47', 'EOG': '#2EAC68', 'Robust_Mixed': '#3B7DD8'}
    labels = {'EMG': 'EMG model', 'EOG': 'EOG model', 'Robust_Mixed': 'Robust Mixed model'}
    dtype_labels_short = {'EMG': 'EMG', 'EOG': 'EOG', 'Mix': 'Mixed', 'Motion': 'Motion'}
    n_types, n_models = len(data_types), len(models)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12),
                                    gridspec_kw={'height_ratios': [1.3, 1]})

    # ═══ (a) Grouped Bar Chart ═══
    x = np.arange(n_types); w = 0.24
    for i, mk in enumerate(models):
        means = [np.mean(results[dt][mk]) for dt in data_types]
        stds  = [np.std(results[dt][mk]) for dt in data_types]
        bars = ax1.bar(x + i * w, means, w, yerr=stds,
                       color=colors[mk], alpha=0.88, edgecolor='white',
                       linewidth=0.6, capsize=3, label=labels[mk], zorder=3)
        for bar, mean in zip(bars, means):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
                     f'{mean:.1f}', ha='center', va='bottom', fontsize=7.5,
                     fontweight='bold', color=colors[mk])

    # ★ mark best per data type
    for j, dt in enumerate(data_types):
        best_mk = max(models, key=lambda m: np.mean(results[dt][m]))
        best_val = np.mean(results[dt][best_mk])
        ax1.annotate('★', xy=(x[j] + models.index(best_mk) * w, best_val + 3),
                     fontsize=26, color=colors[best_mk], ha='center', fontweight='bold')

    ax1.set_xticks(x + w)
    ax1.set_xticklabels([dtype_labels_short[dt] for dt in data_types], fontsize=13)
    ax1.set_ylabel('RMS Amplitude Reduction (%)', fontweight='bold', fontsize=14)
    ax1.set_title('Cross-Model Denoising Performance on Self-Collected Real Artifacts',
                  fontweight='bold', fontsize=16, pad=18)
    ax1.set_ylim(0, 108)
    ax1.legend(loc='lower left', framealpha=0.95, edgecolor='#ccc',
               ncol=3, fontsize=11)
    ax1.grid(axis='y', alpha=0.25, linestyle='--')
    ax1.set_axisbelow(True)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # ═══ (b) Heatmap Matrix ═══
    hm = np.zeros((n_models, n_types))
    for i, mk in enumerate(models):
        for j, dt in enumerate(data_types):
            hm[i, j] = np.mean(results[dt][mk])

    # RdBu style colormap: deep-red(low) → white(mid) → deep-blue(high)
    im = ax2.imshow(hm, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=100)
    ax2.set_xticks(range(n_types))
    ax2.set_xticklabels(['EMG Artifacts', 'EOG Artifacts', 'Mixed Artifacts', 'Motion Artifacts'],
                        fontsize=12)
    ax2.set_yticks(range(n_models))
    ax2.set_yticklabels(['EMG Model', 'EOG Model', 'Robust Mixed Model'], fontsize=12)
    ax2.set_title('Performance Heatmap: RMS Reduction (%)', fontweight='bold', fontsize=15, pad=12)

    for i in range(n_models):
        for j in range(n_types):
            val = hm[i, j]
            tc = 'white' if val < 55 else 'black'
            ax2.text(j, i, f'{val:.1f}%', ha='center', va='center',
                     fontsize=14, fontweight='bold', color=tc)
            if val == np.max(hm[:, j]):
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                     fill=False, edgecolor='black', linewidth=2.5,
                                     linestyle='-')
                ax2.add_patch(rect)

    cbar = plt.colorbar(im, ax=ax2, shrink=0.9, pad=0.02, fraction=0.046)
    cbar.set_label('RMS Reduction (%)', fontweight='bold', fontsize=12)
    cbar.ax.tick_params(labelsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    sp = os.path.join(OUTPUT_DIR, 'cross_summary_figure.png')
    plt.savefig(sp, dpi=350, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {sp}")


if __name__ == '__main__':
    main()
