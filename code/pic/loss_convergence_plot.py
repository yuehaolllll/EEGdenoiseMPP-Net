import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

# 设置全局绘图风格
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['Arial']

# ==================== 1. 配置任务与路径 ====================
configs = [
    {'mode': 'Official', 'task': 'EOG', 'title': 'Task A: EOG Removal'},
    {'mode': 'Official', 'task': 'EMG', 'title': 'Task B: EMG Removal'},
    {'mode': 'SelfMethod', 'task': 'Robust_Mixed', 'title': 'Task C: Mixed Robust (Custom)'}
]

models = {
    'MPP-Net (Ours, Lightweight)': 'MP_Net',
    'MPP-Net (+ Residuals, Heavy)': 'MPR_Net'
}

# 颜色与样式
colors = ['#e74c3c', '#2980b9']
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('Global Convergence Analysis: Full MPR-Net vs. No-Residual Variant', fontsize=18, fontweight='bold')

# ==================== 2. 循环绘制每个任务 ====================
for idx, config in enumerate(configs):
    ax = axes[idx]

    for i, (label, folder) in enumerate(models.items()):
        # 动态拼接路径
        history_path = f'../benchmark_networks/results/{config["mode"]}/{config["task"]}/{folder}/history.npy'

        if not os.path.exists(history_path):
            print(f"⚠️ 缺失文件: {history_path}")
            continue

        history = np.load(history_path, allow_pickle=True).item()
        train_loss = history['loss']['train_mse']
        val_loss = history['loss']['val_mse']
        epochs = range(1, len(train_loss) + 1)

        # 绘制
        ax.plot(epochs, train_loss, label=f'{label} (Train)', color=colors[i], lw=2)
        ax.plot(epochs, val_loss, label=f'{label} (Val)', color=colors[i], linestyle=':', lw=1.5, alpha=0.7)

    # 装饰子图
    ax.set_title(config['title'], fontsize=14, fontweight='bold')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss (MSE)')
    ax.set_yscale('log')  # 对数坐标看细节
    ax.legend(fontsize=9)
    ax.grid(True, which="both", ls="-", alpha=0.4)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('convergence_global_analysis.png', dpi=300)
plt.show()

print("✅ 全场景对比图已生成：convergence_global_analysis.png")