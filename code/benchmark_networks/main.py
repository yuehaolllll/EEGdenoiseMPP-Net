import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import sys
import datetime
import warnings
from thop import profile

# 屏蔽警告
warnings.filterwarnings("ignore", category=UserWarning)

# --- 1. 路径修复与模块导入 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from Network_structure import fcNN, RNN_lstm, Simple_CNN, Complex_CNN
from train_method import train
from loss_function import (
    denoise_loss_rrmset, calculate_cc,
    calculate_spectral_rrmse, get_snr_gain
)

try:
    from Novel_CNN.Novel_CNN import Novel_CNN
    from MP_Net.MP_Net import MPR_Net_ZeroPhase, EEGDenoiseLoss, MPR_Net_SingleScale, MP_Net

    print("✅ 成功加载所有模型架构")
except ImportError as e:
    print(f"⚠️ 部分模块加载失败: {e}")

# ======================== 全自动化配置区 ========================
# 1. 定义要运行的数据模式
DATA_MODES = ['Official', 'SelfMethod']

# 2. 定义每个模式下对应的任务子集
MODE_TASKS = {
    'Official': ['EOG', 'EMG'],
    'SelfMethod': ['Robust_Mixed']
}

# 3. 定义要运行的模型列表
NETWORKS_TO_RUN = ['fcNN',
                   'RNN_lstm',
                   'Simple_CNN',
                   'Complex_CNN',
                   'Novel_CNN',
                   'MPR_Net',             # 带残差
                   'MPR_Net_MSE_Only',    # 消融 1: 全架构 + 纯 MSE
                   'MPR_Net_SingleScale', # 消融 2: 单尺度分支 + 自定义 Loss
                   'MP_Net']              # 主模型无残差

# 4. 硬件配置
DATANUM = 512
BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===============================================================

def get_model_complexity(model_obj, input_size=(1, 1, 512)):
    model_obj.eval()
    dummy_input = torch.randn(input_size).to(DEVICE)
    flops, params = profile(model_obj, inputs=(dummy_input,), verbose=False)
    return f"{params / 1e6:.4f} M", f"{flops / 1e6:.4f} M"


def load_npy(folder, name):
    path = os.path.join(folder, name)
    return torch.from_numpy(np.load(path)).float().unsqueeze(1)


# --- [第一层循环]：数据模式 (Official / SelfMethod) ---
for mode in DATA_MODES:

    # --- [第二层循环]：具体任务 (EOG / EMG / Robust_Mixed) ---
    tasks = MODE_TASKS[mode]
    for task in tasks:

        # 确定当前任务的数据路径
        if mode == 'SelfMethod':
            data_path = './processed_data_SelfMethod/'
        else:
            data_path = f'./processed_data_{task}/'

        print(f"\n{'#' * 60}")
        print(f"📂 当前模式: {mode} | 任务: {task}")
        print(f"📍 数据路径: {data_path}")
        print(f"{'#' * 60}")

        # 加载当前任务的数据集
        try:
            t_x, t_y = load_npy(data_path, 'train_x.npy'), load_npy(data_path, 'train_y.npy')
            v_x, v_y = load_npy(data_path, 'val_x.npy'), load_npy(data_path, 'val_y.npy')
            ts_x, ts_y = load_npy(data_path, 'test_x.npy'), load_npy(data_path, 'test_y.npy')
            train_loader = DataLoader(TensorDataset(t_x, t_y), batch_size=BATCH_SIZE, shuffle=True)
        except Exception as e:
            print(f"❌ 数据加载失败 ({data_path}), 跳过任务。错误: {e}")
            continue

        # --- [第三层循环]：算法模型 ---
        for net_name in NETWORKS_TO_RUN:
            print(f"\n▶️ 启动训练: {net_name} @ {mode}-{task}")

            result_root = f'./results/{mode}/{task}/{net_name}/'
            os.makedirs(result_root, exist_ok=True)

            # 1. 差异化模型与参数配置
            if 'MP' in net_name:  # 进入 MP 系列模型家族逻辑
                current_epochs = 100  # 建议 MP 系列统一跑 100 轮以确保多尺度特征充分收敛
                optimizer_class = optim.AdamW
                lr_rate = 1e-3

                # 情况 A：单尺度消融变体 (用于证明多尺度的必要性)
                if net_name == 'MPR_Net_SingleScale':
                    model = MPR_Net_SingleScale(base_c=24).to(DEVICE)
                    criterion = EEGDenoiseLoss(window_size=DATANUM).to(DEVICE)

                # 情况 B：无残差优化版 (即 MP-Net，极致轻量化版)
                elif net_name == 'MP_Net':
                    model = MP_Net(base_c=24).to(DEVICE)
                    criterion = EEGDenoiseLoss(window_size=DATANUM).to(DEVICE)

                # 情况 C：全功能架构 + 纯 MSE 损失 (用于证明自定义联合 Loss 的必要性)
                elif net_name == 'MPR_Net_MSE_Only':
                    model = MPR_Net_ZeroPhase(base_c=24).to(DEVICE)
                    criterion = nn.MSELoss()

                    # 情况 D：全功能版 (MPR-Net: 多尺度 + 残差 + 联合 Loss，性能上限版)
                elif net_name == 'MPR_Net':
                    model = MPR_Net_ZeroPhase(base_c=24).to(DEVICE)
                    criterion = EEGDenoiseLoss(window_size=DATANUM).to(DEVICE)

                else:
                    raise ValueError(f"未定义的 MP 模型子类型: {net_name}")

                optimizer = optimizer_class(model.parameters(), lr=lr_rate, weight_decay=7e-3)
            else:
                models_dict = {
                    'fcNN': fcNN, 'Simple_CNN': Simple_CNN, 'Complex_CNN': Complex_CNN,
                    'RNN_lstm': RNN_lstm, 'Novel_CNN': Novel_CNN
                }
                model = models_dict[net_name](DATANUM).to(DEVICE)
                optimizer = optim.RMSprop(model.parameters(), lr=0.00005, alpha=0.9)
                criterion = nn.MSELoss()
                current_epochs = 50  # 官方基准50次即可

            # 2. 复杂度统计
            str_p, str_f = get_model_complexity(model, (1, 1, DATANUM))
            print(f"📦 Params: {str_p} | FLOPs: {str_f}")

            # 3. 训练
            history = train(model, train_loader, v_x, v_y, current_epochs, optimizer, DEVICE, result_root, criterion)

            # 4. 评估
            print(f"🔍 正在进行全指标评估...")
            model.load_state_dict(torch.load(os.path.join(result_root, "denoise_model.pth")))
            model.eval()

            # 使用小 Batch 评估防止长久运行后的显存积压
            test_loader = DataLoader(TensorDataset(ts_x, ts_y), batch_size=64, shuffle=False)
            all_preds = []
            with torch.no_grad():
                for tx, _ in test_loader:
                    all_preds.append(model(tx.to(DEVICE)).cpu())

            full_pred = torch.cat(all_preds, dim=0)
            res_rrmse = denoise_loss_rrmset(full_pred, ts_y).item()
            res_cc = calculate_cc(full_pred, ts_y)
            res_spec = calculate_spectral_rrmse(full_pred, ts_y)
            snr_gain, snr_in, snr_out = get_snr_gain(full_pred, ts_y, ts_x)

            # 5. 保存详细报告
            np.save(os.path.join(result_root, 'history.npy'), history)
            curr_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            with open(os.path.join(result_root, 'test_report.txt'), 'w', encoding='utf-8') as f:
                f.write(f"实验报告 - {mode}模式\n")
                f.write(f"模型: {net_name} | 任务: {task} | 时间: {curr_time}\n")
                f.write(f"{'-' * 40}\n")
                f.write(f"1. 复杂度:\n   - Params: {str_p}\n   - FLOPs:  {str_f}\n")
                f.write(f"2. 性能指标:\n   - Temporal RRMSE: {res_rrmse:.4f}\n")
                f.write(f"   - Spectral RRMSE: {res_spec:.4f}\n")
                f.write(f"   - Correlation CC: {res_cc:.4f}\n")
                f.write(f"3. 降噪效果:\n   - SNR Improvement: +{snr_gain:.2f} dB\n")
                f.write(f"   - Input/Output SNR: {snr_in:.2f} / {snr_out:.2f} dB\n")

            print(f"✅ {net_name} 实验完成。报告已存至: {result_root}")

            # 6. 强制清理显存（防止多模型连续运行导致碎片积压）
            del model, optimizer, criterion
            torch.cuda.empty_cache()

print("\n" + "=" * 60)
print("🚀 所有自动化实验（双模式、全任务、全模型）已全部完成！")
print("=" * 60)