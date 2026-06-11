"""
多种子统计评估脚本
对MPP-Net和关键基准模型，用5个随机种子各跑完整训练+测试，
收集 mean±std，生成统计对比表，输出到 pic/statistical_results/
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings("ignore")

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'benchmark_networks'))  # 供 train_method 内部 import

from benchmark_networks.Network_structure import Simple_CNN, Complex_CNN
from benchmark_networks.loss_function import denoise_loss_rrmset, calculate_cc, calculate_spectral_rrmse, get_snr_gain
from benchmark_networks.train_method import train
from Novel_CNN.Novel_CNN import Novel_CNN
from MP_Net.MP_Net import MP_Net, EEGDenoiseLoss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ====== CONFIG ======
SEEDS = [42, 123, 456, 789, 1024]
BATCH_SIZE = 128
DATANUM = 512
OUTPUT_DIR = os.path.join(parent_dir, 'pic', 'statistical_results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Models to evaluate with their configs
MODEL_CONFIGS = {
    'Simple_CNN': {
        'epochs': 50,
        'factory': lambda: Simple_CNN(DATANUM).to(DEVICE),
        'optimizer': lambda model: optim.RMSprop(model.parameters(), lr=0.00005, alpha=0.9),
        'criterion': lambda: nn.MSELoss(),
    },
    'Complex_CNN': {
        'epochs': 50,
        'factory': lambda: Complex_CNN(DATANUM).to(DEVICE),
        'optimizer': lambda model: optim.RMSprop(model.parameters(), lr=0.00005, alpha=0.9),
        'criterion': lambda: nn.MSELoss(),
    },
    'Novel_CNN': {
        'epochs': 50,
        'factory': lambda: Novel_CNN(DATANUM).to(DEVICE),
        'optimizer': lambda model: optim.RMSprop(model.parameters(), lr=0.00005, alpha=0.9),
        'criterion': lambda: nn.MSELoss(),
    },
    'MPP_Net': {
        'epochs': 100,
        'factory': lambda: MP_Net(base_c=24).to(DEVICE),
        'optimizer': lambda model: optim.AdamW(model.parameters(), lr=1e-3, weight_decay=7e-3),
        'criterion': lambda: EEGDenoiseLoss(window_size=DATANUM).to(DEVICE),
    },
}

TASKS = ['EOG', 'EMG']


def load_task_data(task):
    """Load preprocessed data for a given task."""
    data_path = os.path.join(parent_dir, 'benchmark_networks', f'processed_data_{task}')
    t_x = torch.from_numpy(np.load(os.path.join(data_path, 'train_x.npy'))).float().unsqueeze(1)
    t_y = torch.from_numpy(np.load(os.path.join(data_path, 'train_y.npy'))).float().unsqueeze(1)
    v_x = torch.from_numpy(np.load(os.path.join(data_path, 'val_x.npy'))).float().unsqueeze(1)
    v_y = torch.from_numpy(np.load(os.path.join(data_path, 'val_y.npy'))).float().unsqueeze(1)
    ts_x = torch.from_numpy(np.load(os.path.join(data_path, 'test_x.npy'))).float().unsqueeze(1)
    ts_y = torch.from_numpy(np.load(os.path.join(data_path, 'test_y.npy'))).float().unsqueeze(1)
    return t_x, t_y, v_x, v_y, ts_x, ts_y


def evaluate_model(model, ts_x, ts_y):
    """Evaluate a trained model on test set."""
    model.eval()
    test_loader = DataLoader(TensorDataset(ts_x, ts_y), batch_size=64, shuffle=False)
    all_preds = []
    with torch.no_grad():
        for tx, _ in test_loader:
            all_preds.append(model(tx.to(DEVICE)).cpu())
    full_pred = torch.cat(all_preds, dim=0)
    rrmset = denoise_loss_rrmset(full_pred, ts_y).item()
    cc = calculate_cc(full_pred, ts_y)
    rrmsef = calculate_spectral_rrmse(full_pred, ts_y)
    snr_gain, snr_in, snr_out = get_snr_gain(full_pred, ts_y, ts_x)
    return rrmset, cc, rrmsef, snr_gain


def main():
    all_results = {}  # {task: {model: {metric: [val1, val2, ...]}}}

    for task in TASKS:
        print(f"\n{'='*60}")
        print(f"Task: {task}")
        print(f"{'='*60}")

        t_x, t_y, v_x, v_y, ts_x, ts_y = load_task_data(task)
        all_results[task] = {}

        for model_name, config in MODEL_CONFIGS.items():
            print(f"\n  Model: {model_name}")
            model_results = {'rrmset': [], 'cc': [], 'rrmsef': [], 'snr_gain': []}

            for seed in SEEDS:
                print(f"    Seed {seed}...", end=" ")
                torch.manual_seed(seed)
                np.random.seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)

                # Create fresh model
                model = config['factory']()
                optimizer = config['optimizer'](model)
                criterion = config['criterion']()
                epochs = config['epochs']

                # Create DataLoader with seed-controlled shuffle
                train_loader = DataLoader(
                    TensorDataset(t_x, t_y),
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    generator=torch.Generator().manual_seed(seed)
                )

                # Train
                save_dir = os.path.join(OUTPUT_DIR, f'{task}_{model_name}_seed{seed}')
                os.makedirs(save_dir, exist_ok=True)
                train(model, train_loader, v_x, v_y, epochs, optimizer, DEVICE, save_dir, criterion)

                # Load best model and evaluate
                model.load_state_dict(torch.load(os.path.join(save_dir, 'denoise_model.pth'), map_location=DEVICE))
                rrmset, cc, rrmsef, snr_gain = evaluate_model(model, ts_x, ts_y)

                model_results['rrmset'].append(rrmset)
                model_results['cc'].append(cc)
                model_results['rrmsef'].append(rrmsef)
                model_results['snr_gain'].append(snr_gain)
                print(f"RRMSE-t={rrmset:.4f}, CC={cc:.4f}")

                del model, optimizer, criterion
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            all_results[task][model_name] = model_results

    # ===== Generate Report =====
    print("\n" + "=" * 60)
    print("STATISTICAL RESULTS (Mean ± Std over 5 seeds)")
    print("=" * 60)

    report_lines = []
    report_lines.append("多种子统计评估报告 (5 seeds: 42, 123, 456, 789, 1024)")
    report_lines.append("=" * 80)
    report_lines.append("")

    for task in TASKS:
        report_lines.append(f"\n[{task} Task]")
        report_lines.append("-" * 60)
        header = f"{'Model':<16} {'RRMSE-t':>20} {'CC':>18} {'RRMSE-f':>18} {'SNR Gain(dB)':>18}"
        report_lines.append(header)
        report_lines.append("-" * 60)

        for model_name in MODEL_CONFIGS.keys():
            r = all_results[task][model_name]
            rrmset_mean, rrmset_std = np.mean(r['rrmset']), np.std(r['rrmset'])
            cc_mean, cc_std = np.mean(r['cc']), np.std(r['cc'])
            rrmsef_mean, rrmsef_std = np.mean(r['rrmsef']), np.std(r['rrmsef'])
            snr_mean, snr_std = np.mean(r['snr_gain']), np.std(r['snr_gain'])

            line = (f"{model_name:<16} {rrmset_mean:.4f}±{rrmset_std:.4f}  "
                    f"{cc_mean:.4f}±{cc_std:.4f}  {rrmsef_mean:.4f}±{rrmsef_std:.4f}  "
                    f"{snr_mean:.2f}±{snr_std:.2f}")
            report_lines.append(line)
            print(line)

    report_lines.append("\n" + "=" * 80)

    # Save report
    with open(os.path.join(OUTPUT_DIR, 'statistical_report.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    # Save raw data as npz
    result_data = {}
    for task in TASKS:
        for model_name in MODEL_CONFIGS.keys():
            key = f'{task}_{model_name}'
            result_data[key] = all_results[task][model_name]
    np.savez(os.path.join(OUTPUT_DIR, 'raw_results.npz'), **result_data)

    print(f"\nReport saved to: {OUTPUT_DIR}/statistical_report.txt")
    print(f"Raw data saved to: {OUTPUT_DIR}/raw_results.npz")


if __name__ == '__main__':
    main()
