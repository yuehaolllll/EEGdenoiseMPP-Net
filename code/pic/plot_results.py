import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# --- 1. 填入实验数据 ---
models = ['fcNN', 'RNN_lstm', 'Simple_CNN', 'Complex_CNN', 'Novel_CNN', 'MPP_Net']
params = [1.05, 0.79, 16.82, 8.46, 33.56, 0.36] # 单位: M

# EOG 任务数据
eog_rrmse = [0.5409, 0.5539, 0.3612, 0.3559, 0.4206, 0.2990]
eog_cc = [0.8308, 0.8128, 0.9136, 0.9196, 0.8956, 0.9439]

# EMG 任务数据
emg_rrmse = [0.5271, 0.5136, 0.5499, 0.7094, 0.4209, 0.4200]
emg_cc = [0.7946, 0.7999, 0.7486, 0.6852, 0.8617, 0.8628]

# 设置全局样式
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei'] # 用于显示中文，如果环境不支持可注释
plt.rcParams['axes.unicode_minus'] = False

# --- 图表 1: RRMSE 和 CC 的双任务对比图 ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

x = np.arange(len(models))
width = 0.35

# 子图 1: Temporal RRMSE (越低越好)
rects1 = ax1.bar(x - width/2, eog_rrmse, width, label='EOG Task', color='#5DADE2', edgecolor='black')
rects2 = ax1.bar(x + width/2, emg_rrmse, width, label='EMG Task', color='#EC7063', edgecolor='black')
ax1.set_ylabel('Temporal RRMSE (Lower is better)', fontsize=12)
ax1.set_title('A: Denoising Error Comparison', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(models, rotation=30)
ax1.legend()

# 子图 2: Correlation CC (越高越好)
ax2.bar(x - width/2, eog_cc, width, label='EOG Task', color='#5DADE2', edgecolor='black')
ax2.bar(x + width/2, emg_cc, width, label='EMG Task', color='#EC7063', edgecolor='black')
ax2.set_ylabel('Correlation CC (Higher is better)', fontsize=12)
ax2.set_title('B: Waveform Preservation Comparison', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(models, rotation=30)
ax2.set_ylim(0.6, 1.0) # 聚焦高分区间
ax2.legend()

plt.tight_layout()
plt.savefig('performance_comparison.png', dpi=300)

# --- 图表 2: 效率分析图 (散点图: 参数量 vs RRMSE) ---
# 这是一个非常硬核的学术图表，能体现出你模型的极致性价比
plt.figure(figsize=(9, 6))
models_display = ['fcNN', 'RNN_lstm', 'Simple_CNN', 'Complex_CNN', 'Novel_CNN', 'MPP-Net (Ours)']
for i, txt in enumerate(models_display):
    # Ours 用醒目的红色星星，其他用蓝色圆圈
    if 'Ours' in txt:
        plt.scatter(params[i], eog_rrmse[i], s=350, color='#e74c3c', marker='*', edgecolors='black', zorder=5)
        plt.annotate(txt, (params[i], eog_rrmse[i]), xytext=(-40, 15), textcoords='offset points',
                     fontsize=12, fontweight='bold', color='#c0392b')
    else:
        plt.scatter(params[i], eog_rrmse[i], s=150, color='#3498db', alpha=0.7, edgecolors='black')
        plt.annotate(txt, (params[i], eog_rrmse[i]), xytext=(8, 5), textcoords='offset points', fontsize=10)

plt.xscale('log')
plt.xlabel('Model Parameters (M) - Log Scale', fontsize=12, fontweight='bold')
plt.ylabel('Temporal RRMSE (Lower is Better)', fontsize=12, fontweight='bold')
plt.title('Performance vs. Computational Cost (EOG Task)', fontsize=14, fontweight='bold')

# 突出左下角的高效区域
plt.axhspan(0.25, 0.35, xmin=0, xmax=0.3, color='#2ecc71', alpha=0.1)
plt.text(0.12, 0.33, 'Optimal Zone\n(High Fidelity & Lightweight)', color='#27ae60', fontweight='bold', ha='center')

plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig('efficiency_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("图表已生成：performance_comparison.png 和 efficiency_analysis.png")