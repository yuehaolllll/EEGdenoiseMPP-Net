import numpy as np
import random
import os
import math
from scipy import signal


# ==========================================
# --- 内部工具函数 ---
# ==========================================
def resample_data_poly(data, src_fs, tgt_fs):
    if src_fs == tgt_fs: return data
    g = math.gcd(src_fs, tgt_fs)
    up, down = tgt_fs // g, src_fs // g
    return signal.resample_poly(data, up, down)


def get_snr_scale(clean, noise, snr_db):
    rms_clean = np.sqrt(np.mean(clean ** 2))
    rms_noise = np.sqrt(np.mean(noise ** 2))
    if rms_noise < 1e-8: return 0
    return (rms_clean / rms_noise) / (10 ** (snr_db / 20))


def get_signal_segment(subset, target_len, src_fs, tgt_fs):
    raw_sig = subset[random.randint(0, len(subset) - 1)]
    resampled_sig = resample_data_poly(raw_sig, src_fs, tgt_fs)
    if len(resampled_sig) > target_len:
        start = random.randint(0, len(resampled_sig) - target_len)
        return resampled_sig[start: start + target_len]
    elif len(resampled_sig) == target_len:
        return resampled_sig
    else:
        pad_total = target_len - len(resampled_sig)
        return np.pad(resampled_sig, (pad_total // 2, pad_total - pad_total // 2), mode='symmetric')


def add_physio_drift(data, fs):
    t = np.linspace(0, len(data) / fs, len(data))
    drift_amp1 = random.uniform(2.0, 8.0)
    drift_amp2 = random.uniform(1.0, 3.0)
    freq1 = random.uniform(0.05, 0.2)
    freq2 = random.uniform(0.2, 0.5)
    drift = drift_amp1 * np.sin(2 * np.pi * freq1 * t) + drift_amp2 * np.cos(2 * np.pi * freq2 * t)
    return data + drift


def apply_random_gain(data):
    return data * random.uniform(0.8, 1.2)


# ==========================================
# --- 适配 main.py 的主函数 ---
# ==========================================
def prepare_custom_robust_data(EEG_all, noise_all_eog, noise_all_emg, combin_num=1, train_per=0.8, datanum=512):
    """
    适配接口：
    返回: noiseEEG_train, EEG_train, noiseEEG_val, EEG_val, noiseEEG_test, EEG_test, test_std
    """
    SOURCE_FS = 256
    TARGET_FS = 250  # 你的设定
    OUTPUT_LEN = datanum

    # 1. 划分原始数据池
    eeg_split = int(len(EEG_all) * train_per)
    eeg_train_pool = EEG_all[:eeg_split]
    eeg_test_pool = EEG_all[eeg_split:]

    eog_split = int(len(noise_all_eog) * train_per)
    eog_train_pool = noise_all_eog[:eog_split]
    eog_test_pool = noise_all_eog[eog_split:]

    emg_split = int(len(noise_all_emg) * train_per)
    emg_train_pool = noise_all_emg[:emg_split]
    emg_test_pool = noise_all_emg[emg_split:]

    def generate_split(eeg_sub, eog_sub, emg_sub, num_samples):
        X, Y, Scales = [], [], []
        for _ in range(num_samples):
            target = get_signal_segment(eeg_sub, OUTPUT_LEN, SOURCE_FS, TARGET_FS)
            target = apply_random_gain(target)

            # 噪声混合逻辑
            if random.random() < 0.15:  # 纯净信号
                mixed = target.copy()
            else:
                noise_type = random.choices(['EOG', 'EMG', 'COUPLED'], weights=[0.2, 0.4, 0.4])[0]
                if noise_type == 'EOG':
                    noise = get_signal_segment(eog_sub, OUTPUT_LEN, SOURCE_FS, TARGET_FS)
                elif noise_type == 'EMG':
                    noise = get_signal_segment(emg_sub, OUTPUT_LEN, SOURCE_FS, TARGET_FS)
                else:
                    noise = get_signal_segment(eog_sub, OUTPUT_LEN, SOURCE_FS, TARGET_FS) * random.uniform(0.5, 1.5) + \
                            get_signal_segment(emg_sub, OUTPUT_LEN, SOURCE_FS, TARGET_FS) * random.uniform(0.5, 1.5)

                scale = get_snr_scale(target, noise, random.uniform(-25.0, 10.0))
                mixed = target + noise * scale

            mixed = add_physio_drift(mixed, TARGET_FS)
            if random.random() > 0.5: mixed += np.random.normal(0, random.uniform(0.01, 0.05), OUTPUT_LEN)

            # 归一化
            rms_val = np.sqrt(np.mean(mixed ** 2))
            norm_factor = rms_val if rms_val > 1e-6 else 1.0

            X.append(np.clip(mixed / norm_factor, -5.0, 5.0))
            Y.append(np.clip(target / norm_factor, -5.0, 5.0))
            Scales.append(norm_factor)

        return np.array(X), np.array(Y), np.array(Scales)

    print("正在生成自定义增强数据集...")
    # 数量可以根据需要调整，这里建议训练 40000，验证 5000，测试 5000
    train_x, train_y, _ = generate_split(eeg_train_pool, eog_train_pool, emg_train_pool, 40000)
    val_x, val_y, _ = generate_split(eeg_test_pool, eog_test_pool, emg_test_pool, 5000)
    test_x, test_y, test_std = generate_split(eeg_test_pool, eog_test_pool, emg_test_pool, 5000)

    return train_x, train_y, val_x, val_y, test_x, test_y, test_std


if __name__ == "__main__":
    # 1. 路径配置 (请确保路径正确)
    RAW_DATA_DIR = r'F:\EEG\EEGdenoiseNet\EEGdenoiseNet_pytorch\data'
    SAVE_DIR = r'./processed_data_SelfMethod'  # 存放在当前目录下
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("正在读取原始 npy 文件...")
    eeg = np.load(os.path.join(RAW_DATA_DIR, 'EEG_all_epochs.npy'))
    eog = np.load(os.path.join(RAW_DATA_DIR, 'EOG_all_epochs.npy'))
    emg = np.load(os.path.join(RAW_DATA_DIR, 'EMG_all_epochs.npy'))

    # 2. 生成数据 (按照你之前的 40000, 5000, 5000 规模)
    # 这里直接调用你定义的生成逻辑
    t_x, t_y, v_x, v_y, ts_x, ts_y, ts_std = prepare_custom_robust_data(eeg, eog, emg, datanum=512)

    # 3. 保存为 npy 文件
    print(f"正在保存数据至 {SAVE_DIR} ...")
    np.save(os.path.join(SAVE_DIR, 'train_x.npy'), t_x)
    np.save(os.path.join(SAVE_DIR, 'train_y.npy'), t_y)
    np.save(os.path.join(SAVE_DIR, 'val_x.npy'), v_x)
    np.save(os.path.join(SAVE_DIR, 'val_y.npy'), v_y)
    np.save(os.path.join(SAVE_DIR, 'test_x.npy'), ts_x)
    np.save(os.path.join(SAVE_DIR, 'test_y.npy'), ts_y)
    np.save(os.path.join(SAVE_DIR, 'test_std.npy'), ts_std)

    print("✅ 自定义数据集预生成完成！")