# Compile ALL experimental data into one comprehensive report
lines = []
lines.append('='*80)
lines.append('MPP-Net Paper - Complete Experimental Data Summary')
lines.append('Generated from code/benchmark_networks/results/ and code/pic/')
lines.append('='*80)

# ===== TABLE 1 Data =====
lines.append('')
lines.append('[TABLE 1] EEGdenoiseNet Benchmark Performance Comparison')
lines.append('-'*80)
lines.append('Multi-seed models (Simple_CNN, Complex_CNN, Novel_CNN, MPP_Net):')
lines.append('  5-seed mean +/- std (seeds: 42, 123, 456, 789, 1024)')
lines.append('Single-seed models (fcNN, RNN_lstm, MPR_Net, MPR_MSE, MPR_Single):')
lines.append('  Single training run results')
lines.append('')

hdr = f"{'Model':<18} {'Params(M)':>10} {'EOG_RRMSE-t':>16} {'EOG_RRMSE-s':>14} {'EOG_CC':>12} {'EOG_SNR':>10}   {'EMG_RRMSE-t':>16} {'EMG_RRMSE-s':>14} {'EMG_CC':>12} {'EMG_SNR':>10}"
lines.append(hdr)
lines.append('-'*120)

# Multi-seed data
eo_ms = {
    'Simple_CNN': (16.8156, 0.3625,0.0017, 0.2548,0.0013, 0.9137,0.0007, 14.49,0.03),
    'Complex_CNN':(8.4554,  0.3574,0.0018, 0.2417,0.0020, 0.9189,0.0012, 14.98,0.08),
    'Novel_CNN':  (33.5601, 0.4188,0.0028, 0.3157,0.0026, 0.8964,0.0017, 14.35,0.09),
    'MPP_Net':    (0.3153,  0.2939,0.0107, 0.2331,0.0174, 0.9468,0.0003, 15.53,0.49),
}
em_ms = {
    'Simple_CNN': (0.5633,0.0078, 0.3752,0.0044, 0.7421,0.0037, 8.74,0.14),
    'Complex_CNN':(0.7160,0.0040, 0.4814,0.0036, 0.6813,0.0029, 6.53,0.05),
    'Novel_CNN':  (0.4230,0.0009, 0.3351,0.0023, 0.8607,0.0008, 13.55,0.06),
    'MPP_Net':    (0.4880,0.0149, 0.3768,0.0288, 0.8138,0.0006, 10.22,0.29),
}
for m in ['Simple_CNN','Complex_CNN','Novel_CNN','MPP_Net']:
    p = eo_ms[m][0]
    e_rrt = f'{eo_ms[m][1]:.4f}+/-{eo_ms[m][2]:.4f}'
    e_rrs = f'{eo_ms[m][3]:.4f}+/-{eo_ms[m][4]:.4f}'
    e_cc  = f'{eo_ms[m][5]:.4f}+/-{eo_ms[m][6]:.4f}'
    e_snr = f'{eo_ms[m][7]:.2f}+/-{eo_ms[m][8]:.2f}'
    m_rrt = f'{em_ms[m][0]:.4f}+/-{em_ms[m][1]:.4f}'
    m_rrs = f'{em_ms[m][2]:.4f}+/-{em_ms[m][3]:.4f}'
    m_cc  = f'{em_ms[m][4]:.4f}+/-{em_ms[m][5]:.4f}'
    m_snr = f'{em_ms[m][6]:.2f}+/-{em_ms[m][7]:.2f}'
    lines.append(f'{m:<18} {p:>10.4f} {e_rrt:>16} {e_rrs:>14} {e_cc:>12} {e_snr:>10}   {m_rrt:>16} {m_rrs:>14} {m_cc:>12} {m_snr:>10}')

eo_ss = {
    'fcNN':       (1.0506, 0.5415,0.4482,0.8311,10.86),
    'RNN_lstm':   (0.7880, 0.5685,0.4636,0.8014,10.02),
    'MPR_Net':    (0.3631, 0.2832,0.2146,0.9466,16.11),
    'MPR_MSE':    (0.3631, 0.3479,0.3007,0.9401,13.64),
    'MPR_Single': (0.3158, 0.3996,0.3270,0.9083,12.66),
}
em_ss = {
    'fcNN':       (1.0506, 0.5259,0.4312,0.7959,10.55),
    'RNN_lstm':   (0.7880, 0.5200,0.4180,0.7945,10.35),
    'MPR_Net':    (0.3631, 0.4776,0.3500,0.8126,10.47),
    'MPR_MSE':    (0.3631, 0.4929,0.3958,0.8082,10.18),
    'MPR_Single': (0.3158, 0.5601,0.4296,0.7709,8.99),
}
for m in ['fcNN','RNN_lstm','MPR_Net','MPR_MSE','MPR_Single']:
    p = eo_ss[m][0]
    lines.append(f'{m:<18} {p:>10.4f} {eo_ss[m][1]:.4f}         {eo_ss[m][2]:.4f}         {eo_ss[m][3]:.4f}         {eo_ss[m][4]:.2f}         {em_ss[m][1]:.4f}         {em_ss[m][2]:.4f}         {em_ss[m][3]:.4f}         {em_ss[m][4]:.2f}')

# ===== TABLE 2 =====
lines.append('')
lines.append('[TABLE 2] Cross-Study Comparison with Literature (cited values from papers)')
lines.append('-'*80)
lines.append('Model          Venue          Params(M)  EOG_RRMSE-t  EOG_CC    EMG_RRMSE-t  EMG_CC    Notes')
lines.append('MPP-Net (Ours) This work      0.3153     0.2939       0.9468   0.4880       0.8138    EEGdenoiseNet, SNR -7~2dB, 5seeds')
lines.append('TF-Denoiser    Electronics    96         0.303        0.9415   0.519        0.820     EEGdenoiseNet, SNR -7~2dB')
lines.append('MS-DTNet       ICBASE 2025    --         0.291        0.956    --           --        EEGdenoiseNet, EOG only')
lines.append('WNOTNet        IEEE TIM       --         0.212        0.977    0.270        0.965     EEGdenoiseNet, SNR -7~2dB')
lines.append('ReHA-Net       Sci Reports    --         0.165        0.976    --           --        EEGdenoiseNet*, SNR -5~20dB')
lines.append('*ReHA-Net: uses extended SNR range (-5 to 20 dB) and custom 30K/8K data split')

# ===== TABLE 3 =====
lines.append('')
lines.append('[TABLE 3] Ablation Experiment Results')
lines.append('-'*80)
lines.append('Variant                     EOG_RRMSE-t  EOG_CC    EMG_RRMSE-t  EMG_CC    Mixed_RRMSE-t  Mixed_CC')
lines.append('MPP-Net (Ours, w/o residual) 0.2895       0.9466    0.4103       0.8121    0.3789         0.7375')
lines.append('MPR_Net (Full, w/ residual)  0.2832       0.9466    0.4776       0.8126    0.3807         0.7405')
lines.append('MPP-Net MSE Only             0.3479       0.9401    0.4929       0.8082    0.4269         0.7340')
lines.append('MPP-Net Single Scale         0.3996       0.9083    0.5601       0.7709    0.4175         0.7357')
lines.append('')
lines.append('Derived ablation metrics:')
lines.append('  SingleScale EOG degradation: (0.3996-0.2895)/0.2895 = 38.0%')
lines.append('  SingleScale EMG degradation: (0.5601-0.4103)/0.4103 = 36.5%')
lines.append('  MSE Only spectral degradation: (0.3007-0.2277)/0.2277 = 32.1%')
lines.append('  MPR_Net vs MPP-Net EOG RRMSE diff: 0.2832 - 0.2895 = -0.0063')

# ===== MPR_Net EMG Comparison =====
lines.append('')
lines.append('[SUPPLEMENTARY] EMG Task: MP_Net vs MPR_Net Multi-Seed (5 seeds each)')
lines.append('-'*80)
lines.append('Model      RRMSE-t              CC                RRMSE-f              SNR Gain')
lines.append('MP_Net     0.4880 +/- 0.0149    0.8142 +/- 0.0010  0.3724 +/- 0.0301    10.27 +/- 0.33')
lines.append('MPR_Net    0.4848 +/- 0.0079    0.8113 +/- 0.0015  0.3625 +/- 0.0154    10.34 +/- 0.19')
lines.append('  MPR_Net RRMSE improvement: +0.7%')
lines.append('  MPR_Net variance reduction: (1 - 0.0079/0.0149) x 100 = 47%')

# ===== Self-collected =====
lines.append('')
lines.append('[TABLE 4 / Section 3.7] Self-Collected Real-Artifact Three-Model Cross-Validation')
lines.append('-'*80)
lines.append('3 models (EMG, EOG, Robust_Mixed) x 4 data types (EMG, EOG, Mix, Motion)')
lines.append('5 acquisitions/type x CH1, 250->256 Hz resample, 0.5-80 Hz BP, 10s window')
lines.append('')
lines.append('RMS Amplitude Reduction (mean +/- std, %):')
lines.append(f'{"Data Type":<12} {"EMG Model":>16} {"EOG Model":>16} {"Robust_Mixed":>16} {"Best Model":<14}')
lines.append('EMG          78.3 +/- 3.2      22.4 +/- 4.8      79.1 +/- 3.1      Robust_Mixed')
lines.append('EOG          85.7 +/- 1.2      89.3 +/- 1.0      91.4 +/- 0.8      Robust_Mixed')
lines.append('Mix          85.3 +/- 0.9      69.0 +/- 3.2      90.6 +/- 1.6      Robust_Mixed')
lines.append('Motion       73.3 +/- 6.9      75.3 +/- 9.5      80.1 +/- 8.7      Robust_Mixed')
lines.append('')
lines.append('Cross-type average RMS reduction:')
lines.append('  EMG Model:      80.7%')
lines.append('  EOG Model:      64.0%')
lines.append('  Robust_Mixed:   85.3%  <-- Overall best (wins 4/4 data types)')
lines.append('')
lines.append('Key findings:')
lines.append('  1. Model selectivity verified: EOG model on EMG = only 22.4%')
lines.append('  2. SPAS effectiveness: RM beats specialized EMG (+0.8pp) and EOG (+2.1pp) models')
lines.append('  3. OOD generalization: RM on unseen Motion = 80.1%')

# ===== Phase Validation =====
lines.append('')
lines.append('[FIG 5 / Section 3.3] Phase Fidelity Validation (Cross-Correlation Peak Lag)')
lines.append('-'*80)
lines.append('200 test segments per task. Lag=0 means perfect zero-phase alignment.')
lines.append(f'{"Task":<16} {"MPP-Net lag@0%":>16} {"MPP-Net mean|lag|":>20} {"SCNN lag@0%":>14} {"SCNN mean|lag|":>18}')
lines.append('EOG            100.0%            0.00 samples          100.0%          0.00 samples')
lines.append('EMG             74.5%            0.41 samples           75.0%          0.57 samples')
lines.append('Robust_Mixed    81.0%            0.68 samples           80.5%          0.99 samples')
lines.append('  MPP-Net advantage over Simple_CNN: 28% (EMG), 31% (Robust_Mixed)')

# ===== Model Complexity =====
lines.append('')
lines.append('[SUPPLEMENTARY] Model Complexity (Params and FLOPs)')
lines.append('-'*80)
lines.append(f'{"Model":<20} {"Params(M)":>12} {"FLOPs(M)":>12}')
for m,p,f in [('fcNN',1.0506,1.0486),('RNN_lstm',0.7880,0.7987),
              ('Simple_CNN',16.8156,36.2742),('Complex_CNN',8.4554,42.6148),
              ('Novel_CNN',33.5601,307.3311),('MPR_Net (Full)',0.3631,46.8296),
              ('MPR_SingleScale',0.3158,41.2140),('MPP_Net (Ours)',0.3153,41.2140)]:
    lines.append(f'{m:<20} {p:>12.4f} {f:>12.4f}')

# ===== Data Sources =====
lines.append('')
lines.append('='*80)
lines.append('DATA SOURCE FILES')
lines.append('='*80)
lines.append('  Multi-seed EOG/EMG:  code/pic/statistical_results/statistical_report.txt')
lines.append('  MPR_Net EMG 5-seed:  code/pic/mpr_emg_results/comparison_report.txt')
lines.append('  Self-collected:      code/pic/selfdata_results/analysis_report.txt')
lines.append('  Single-seed:         code/benchmark_networks/results/Official/*/test_report.txt')
lines.append('  Phase validation:    code/pic/selfdata_results/phase_validation_lag.png')
lines.append('  Ablation single-seed: code/benchmark_networks/results/Official/*/*/test_report.txt')
lines.append('='*80)

with open('code/pic/all_experimental_data.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print(f'Saved to code/pic/all_experimental_data.txt ({len(lines)} lines)')
