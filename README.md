# MPP-Net: Multi-scale Phase-Preserving Network for Single-Channel EEG Artifact Removal

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Official PyTorch implementation of **MPP-Net**, a high-fidelity lightweight deep learning model for removing physiological artifacts (EOG, EMG, composite) from single-channel EEG signals.

> 📄 **Paper**: *A High-Fidelity Lightweight Multi-Scale Phase-Preserving Network for Single-Channel EEG Artifact Removal* (under review)

---

## 🔑 Key Features

- **Ultra-lightweight**: Only **0.31M parameters**, suitable for wearable and edge devices
- **Zero-phase design**: Symmetric padding ensures mathematically guaranteed phase preservation
- **Three-domain joint loss**: Temporal Smooth L1 + Spectral log-power consistency + Pearson correlation
- **Multi-scale architecture**: Four parallel dilated convolution branches (rates 1, 4, 8)
- **SPAS strategy**: Synergistic Physiological Artifact Simulation for composite artifact training
- **Multi-seed evaluation**: Statistical validation with mean ± std across 5 random seeds
- **Self-collected real-data validation**: Tested on custom hardware with dry electrodes

---

## 📁 Repository Structure

```
code/
├── MP_Net/                          # MPP-Net model definition
│   └── MP_Net.py                    # MP_Net, MPR_Net, EEGDenoiseLoss, ablation variants
│
├── Novel_CNN/                       # Novel_CNN baseline (Sun et al.)
│   └── Novel_CNN.py
│
├── benchmark_networks/              # Core training & evaluation pipeline
│   ├── main.py                      # Main entry: automated multi-model/multi-task training
│   ├── train_method.py              # Training loop with validation
│   ├── loss_function.py             # Evaluation metrics (RRMSE, CC, SNR)
│   ├── Network_structure.py         # Baseline models (fcNN, RNN, Simple_CNN, Complex_CNN)
│   ├── data_prepare.py              # EEGdenoiseNet data preparation (EMG/EOG)
│   ├── data_prepare_selfmethod.py   # SPAS strategy data preparation (Robust_Mixed)
│   ├── benchmark_latency.py         # Inference latency benchmarking
│   ├── processed_data_EMG/          # Preprocessed EMG data (.npy)
│   ├── processed_data_EOG/          # Preprocessed EOG data (.npy)
│   ├── processed_data_SelfMethod/   # Preprocessed Robust_Mixed data (.npy)
│   └── results/                     # Training results & model checkpoints
│
├── selfdata/                        # Self-collected real-artifact data
│   ├── EMG/                         # 5 EMG recordings (250 Hz, CSV)
│   ├── EOG/                         # 5 EOG recordings (250 Hz, CSV)
│   ├── Mix/                         # 5 mixed artifact recordings
│   └── Motion/                      # 5 motion artifact recordings
│
└── pic/                             # Visualization & analysis scripts
    ├── plot_paper_figures.py         # Combined Fig 3 (SNR), Fig 4 (Waveform+PSD), Fig 5 (Efficiency)
    ├── plot_phase_validation.py      # Cross-correlation phase fidelity analysis
    ├── selfdata_analysis.py          # Self-collected data 3-model cross-validation
    ├── multi_seed_eval.py            # Multi-seed statistical evaluation
    ├── mpr_emg_test.py              # MP_Net vs MPR_Net EMG comparison
    ├── plot_cross_summary.py         # Cross-validation summary heatmap
    ├── plot_denoising_gallery.py     # 4-type denoising effect gallery
    ├── compile_data.py              # Compile all experimental data into summary
    └── selfdata_results/            # Generated figures & reports
```

---

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch numpy scipy matplotlib
```

### 1. Prepare Training Data

Download the EEGdenoiseNet dataset and place `.npy` files in `data/`:

```bash
# Expected files:
#   data/EEG_all_epochs.npy    # Clean EEG (4514 segments x 512 samples)
#   data/EMG_all_epochs.npy    # EMG artifacts (5598 segments)
#   data/EOG_all_epochs.npy    # EOG artifacts (3400 segments)
```

Generate preprocessed data:

```bash
cd code/benchmark_networks
python data_prepare.py           # EMG/EOG tasks
python data_prepare_selfmethod.py # Robust_Mixed task (SPAS strategy)
```

### 2. Train Models

```bash
cd code/benchmark_networks
python main.py
```

This will automatically run all 9 model variants on EMG, EOG, and Robust_Mixed tasks. Results are saved to `results/`.

### 3. Evaluate & Visualize

```bash
cd code/pic

# Generate paper figures (Fig 3, 4, 5)
python plot_paper_figures.py

# Phase fidelity validation (cross-correlation lag)
python plot_phase_validation.py

# Self-collected data cross-validation
python selfdata_analysis.py

# Multi-seed statistical evaluation (longer runtime)
python multi_seed_eval.py
```

---

## 🏗️ Model Architecture

MPP-Net adopts a U-Net-like symmetric encoder-decoder architecture:

```
Input (1 x 512)
  │
  ├─ ReVIN Normalization
  ├─ Encoder Stage 1: MPRB (24 ch) → ZeroPhasePool
  ├─ Encoder Stage 2: MPRB (48 ch) → ZeroPhasePool
  ├─ Encoder Stage 3: MPRB (96 ch) → ZeroPhasePool
  ├─ Bottleneck: MPRB (192 ch)
  ├─ Decoder Stage 3: Upsample + MPRB (96 ch)
  ├─ Decoder Stage 2: Upsample + MPRB (48 ch)
  ├─ Decoder Stage 1: Upsample + MPRB (24 ch)
  └─ Output Conv (1 ch)
```

**MPRB (Multi-scale Parallel Residual Block):**

```
Input ─┬─ Conv1d(K=7, D=8) ─┐
       ├─ Conv1d(K=5, D=4) ─┤
       ├─ Conv1d(K=3, D=1) ─┼─ Concat ─ BN ─ PReLU ─ Dropout ─ Output
       └─ Conv1d(K=1)      ─┘
```

All convolutions use **symmetric padding** (zero-phase) to preserve temporal alignment.

---

## 📊 Key Results

| Model | Params | EOG RRMSE-t ↓ | EOG CC ↑ | EMG RRMSE-t ↓ | EMG CC ↑ |
|-------|:------:|:------------:|:--------:|:------------:|:--------:|
| fcNN | 1.05M | 0.5415 | 0.8311 | 0.5259 | 0.7959 |
| Simple_CNN | 16.82M | 0.3625±0.0017 | 0.9137±0.0007 | 0.5633±0.0078 | 0.7421±0.0037 |
| Complex_CNN | 8.46M | 0.3574±0.0018 | 0.9189±0.0012 | 0.7160±0.0040 | 0.6813±0.0029 |
| Novel_CNN | 33.56M | 0.4188±0.0028 | 0.8964±0.0017 | 0.4230±0.0009 | 0.8607±0.0008 |
| **MPP-Net** | **0.31M** | **0.2939±0.0107** | **0.9468±0.0003** | 0.4880±0.0149 | 0.8138±0.0006 |

*Multi-seed values shown as mean ± std (5 seeds). Single-seed values for non-multi-seed models.*

**Self-collected data** (Robust_Mixed model): **85.3% average RMS reduction** across 4 artifact types, wins 4/4.

---

## 📝 Citation

```bibtex
@article{mppnet2025,
  title={A High-Fidelity Lightweight Multi-Scale Phase-Preserving Network for Single-Channel EEG Artifact Removal},
  author={},
  journal={Biomedical Signal Processing and Control},
  year={2025},
  note={Under review}
}
```

## 📄 License

This project is released under the MIT License.

---

*For questions or collaboration, please open an issue or contact the authors.*
