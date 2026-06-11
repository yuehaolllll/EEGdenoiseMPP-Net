"""
MPR_Net 在 EMG 任务上的多种子测试
对比 MPR_Net (带残差) vs MP_Net (无残差) 在 EMG 上的稳定性和性能
"""
import torch, torch.nn as nn, torch.optim as optim, numpy as np, os, sys, warnings
from torch.utils.data import DataLoader, TensorDataset
warnings.filterwarnings("ignore")

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'benchmark_networks'))

from benchmark_networks.loss_function import denoise_loss_rrmset, calculate_cc, calculate_spectral_rrmse, get_snr_gain
from benchmark_networks.train_method import train
from MP_Net.MP_Net import MP_Net, MPR_Net_ZeroPhase, EEGDenoiseLoss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [42, 123, 456, 789, 1024]
BATCH_SIZE, DATANUM, EPOCHS = 128, 512, 100
OUTPUT_DIR = os.path.join(parent_dir, 'pic', 'mpr_emg_results')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    path = os.path.join(parent_dir, 'benchmark_networks', 'processed_data_EMG')
    t_x = torch.from_numpy(np.load(os.path.join(path, 'train_x.npy'))).float().unsqueeze(1)
    t_y = torch.from_numpy(np.load(os.path.join(path, 'train_y.npy'))).float().unsqueeze(1)
    v_x = torch.from_numpy(np.load(os.path.join(path, 'val_x.npy'))).float().unsqueeze(1)
    v_y = torch.from_numpy(np.load(os.path.join(path, 'val_y.npy'))).float().unsqueeze(1)
    ts_x = torch.from_numpy(np.load(os.path.join(path, 'test_x.npy'))).float().unsqueeze(1)
    ts_y = torch.from_numpy(np.load(os.path.join(path, 'test_y.npy'))).float().unsqueeze(1)
    return t_x, t_y, v_x, v_y, ts_x, ts_y


def evaluate(model, ts_x, ts_y):
    model.eval()
    loader = DataLoader(TensorDataset(ts_x, ts_y), batch_size=64, shuffle=False)
    preds = []
    with torch.no_grad():
        for tx, _ in loader:
            preds.append(model(tx.to(DEVICE)).cpu())
    full = torch.cat(preds, dim=0)
    return (denoise_loss_rrmset(full, ts_y).item(),
            calculate_cc(full, ts_y),
            calculate_spectral_rrmse(full, ts_y),
            get_snr_gain(full, ts_y, ts_x)[0])


def run_one(model_name, factory, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)

    model = factory().to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=7e-3)
    criterion = EEGDenoiseLoss(window_size=DATANUM).to(DEVICE)
    loader = DataLoader(TensorDataset(t_x, t_y), batch_size=BATCH_SIZE, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))

    save_dir = os.path.join(OUTPUT_DIR, f'{model_name}_seed{seed}')
    os.makedirs(save_dir, exist_ok=True)
    train(model, loader, v_x, v_y, EPOCHS, opt, DEVICE, save_dir, criterion)

    model.load_state_dict(torch.load(os.path.join(save_dir, 'denoise_model.pth'), map_location=DEVICE))
    rrmset, cc, rrmsef, snr = evaluate(model, ts_x, ts_y)
    del model, opt, criterion
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return rrmset, cc, rrmsef, snr


t_x, t_y, v_x, v_y, ts_x, ts_y = load_data()
print(f"Device: {DEVICE} | EMG data: train={len(t_x)}, val={len(v_x)}, test={len(ts_x)}")

configs = [
    ('MP_Net', lambda: MP_Net(base_c=24)),
    ('MPR_Net', lambda: MPR_Net_ZeroPhase(base_c=24)),
]

all_results = {}

for name, factory in configs:
    print(f"\n{'='*50}\n{name} on EMG\n{'='*50}")
    r_ = {'rrmset': [], 'cc': [], 'rrmsef': [], 'snr': []}
    for seed in SEEDS:
        print(f"  Seed {seed}...", end=" ")
        rrmset, cc, rrmsef, snr = run_one(name, factory, seed)
        r_['rrmset'].append(rrmset); r_['cc'].append(cc)
        r_['rrmsef'].append(rrmsef); r_['snr'].append(snr)
        print(f"RRMSE-t={rrmset:.4f}  CC={cc:.4f}  SNR={snr:+.2f}")
    all_results[name] = r_

# Report
print("\n" + "=" * 60)
print("EMG 任务: MP_Net vs MPR_Net (5 seeds)")
print("=" * 60)
for name in ['MP_Net', 'MPR_Net']:
    r = all_results[name]
    print(f"\n{name}:")
    for metric, label in [('rrmset', 'RRMSE-t'), ('cc', 'CC'), ('rrmsef', 'RRMSE-f'), ('snr', 'SNR gain')]:
        vals = r[metric]
        print(f"  {label}: {np.mean(vals):.4f} ± {np.std(vals):.4f}   [{', '.join(f'{v:.4f}' for v in vals)}]")

# Compare
mp = all_results['MP_Net']['rrmset']
mr = all_results['MPR_Net']['rrmset']
improvement = (np.mean(mp) - np.mean(mr)) / np.mean(mp) * 100
print(f"\nMPR_Net vs MP_Net RRMSE improvement: {improvement:+.1f}%")
print(f"MP_Net  std: {np.std(mp):.4f}  |  MPR_Net std: {np.std(mr):.4f}")

with open(os.path.join(OUTPUT_DIR, 'comparison_report.txt'), 'w') as f:
    f.write(f"EMG MP_Net vs MPR_Net (5 seeds)\n{'='*40}\n")
    for name in ['MP_Net', 'MPR_Net']:
        r = all_results[name]
        f.write(f"\n{name}:\n")
        for m, l in [('rrmset','RRMSE-t'),('cc','CC'),('rrmsef','RRMSE-f'),('snr','SNR')]:
            v = r[m]
            f.write(f"  {l}: {np.mean(v):.4f}±{np.std(v):.4f}\n")
    f.write(f"\nMPR_Net improves RRMSE by {improvement:+.1f}%\n")
