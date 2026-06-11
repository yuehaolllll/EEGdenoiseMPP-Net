import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import re

# ==================== 1. 配置与路径定义 ====================
# 设置全局风格
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['Arial']

# 路径定位
results_base = '../benchmark_networks/results'

# 任务配置
tasks = [
    {'mode': 'Official', 'task': 'EOG', 'title': 'Task: EOG (Temporal)'},
    {'mode': 'Official', 'task': 'EMG', 'title': 'Task: EMG (Temporal)'},
    {'mode': 'SelfMethod', 'task': 'Robust_Mixed', 'title': 'Task: Mixed Robust (Temporal)'}
]

# 消融变体对应文件夹名 (注意顺序)
variants_map = [
    ('MPP-Net\n(Ours, Final)', 'MP_Net'),               # 把你的最终模型放在最前面或最后面
    ('MPP-Net\n(+ Residuals)', 'MPR_Net'),              # 证明残差冗余
    ('MPP-Net\n(Single-Scale)', 'MPR_Net_SingleScale'), # 证明多尺度必要
    ('MPP-Net\n(MSE Only)', 'MPR_Net_MSE_Only')         # 证明联合损失必要
]


# ==================== 2. 自动化数据抓取函数 ====================
def fetch_metric(mode, task, model_folder):
    report_path = os.path.join(results_base, mode, task, model_folder, 'test_report.txt')
    if not os.path.exists(report_path):
        print(f"⚠️ 找不到报告: {report_path}")
        return 0.0, 0.0

    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()
        # 使用正则匹配数值
        rrmse = float(re.findall(r'Temporal RRMSE.*?: ([\d.]+)', content)[0])
        spec_rrmse = float(re.findall(r'Spectral RRMSE.*?: ([\d.]+)', content)[0])
        return rrmse, spec_rrmse


# ==================== 3. 执行绘图逻辑 ====================
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle('Ablation Study: Impact of Multi-Scale, Residuals, and Joint Loss', fontsize=20, fontweight='bold')

colors = ['#c0392b', '#2980b9', '#27ae60', '#7f8c8d'] # 红色突出Ours

for idx, config in enumerate(tasks):
    ax = axes[idx]
    rrmse_vals = []
    labels = []

    for label, folder in variants_map:
        val, _ = fetch_metric(config['mode'], config['task'], folder)
        rrmse_vals.append(val)
        labels.append(label)

    # 绘制柱状图
    bars = ax.bar(labels, rrmse_vals, color=colors, edgecolor='black', alpha=0.85, width=0.6)
    ax.set_title(config['title'], fontsize=15, fontweight='bold')
    ax.set_ylabel('RRMSE (Lower is Better)', fontsize=12)

    # 动态调整y轴刻度，让对比更明显
    min_v, max_v = min(rrmse_vals), max(rrmse_vals)
    ax.set_ylim(min_v * 0.8, max_v * 1.1)

    # 标注数值
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('ablation_performance_final.png', dpi=300)

# ==================== 4. 专项 Loss 效能对比 (以 EOG 为例) ====================
plt.figure(figsize=(8, 6))
full_spec, _ = fetch_metric('Official', 'EOG', 'MPR_Net')
# 注意：fetch_metric 返回的是 (Temporal, Spectral)，我们要的是第二个值
_, full_spec = fetch_metric('Official', 'EOG', 'MPR_Net')
_, mse_spec = fetch_metric('Official', 'EOG', 'MPR_Net_MSE_Only')

spec_data = [full_spec, mse_spec]
spec_labels = ['MPR-Net\n(Joint Loss)', 'MPR-Net\n(MSE Only)']

plt.bar(spec_labels, spec_data, color=['#e74c3c', '#2c3e50'], edgecolor='black', width=0.4)
plt.ylabel('Spectral RRMSE (Lower is Better)', fontsize=12)
plt.title('Impact of Multi-Objective Loss on Spectral Fidelity (EOG)', fontsize=14, fontweight='bold')

improvement = (mse_spec - full_spec) / mse_spec * 100
plt.text(0.5, max(spec_data) * 0.9, f'Spectral Distortion Reduced by {improvement:.1f}%',
         ha='center', color='darkred', fontweight='bold', fontsize=12)

plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('ablation_spectral_loss.png', dpi=300)

plt.show()