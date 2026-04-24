# CG-LSTM for HRRP Pose Estimation

This repository contains the PyTorch implementation of the paper's
correlation-guided LSTM framework for space-target pose estimation from short
HRRP sequences.

## Method Coverage

The open-source training path follows the method described in the manuscript:

- ASC proxy extraction from sparse OMP HRRPs.
- Adjacent-profile amplitude correlation `S_{t,t-1}` using range/amplitude
  Gaussian kernels and Hungarian matching.
- Adjacent-profile phase correlation `Z_{t,t-1}` using the same matched ASC set
  and circular phase consistency.
- Two-stage fusion: GMU fusion of `S/Z`, followed by scaled dot-product
  cross-attention with the RLOS increment prior.
- CG-LSTM gate modulation with the fused feature `Gamma_{t,t-1}`.
- Unit-vector regression loss to reduce angular wrap-around effects.

## Data Layout

The default layout is:

```text
code/data/
  attitute_ground_truth.json
  test/
    attitude_1/*.mat
    attitude_2/*.mat
    attitude_3/*.mat
```

Each `.mat` file is expected to contain:

- `hrrp`: HRRP sequence with shape `[T, D]`.
- `hrrp_omp`: sparse HRRP/ASC proxy with shape `[T, D]`.
- `hrrp_phase`: phase sequence with shape `[T, D]`.
- `theta_phi`: RLOS angles in degrees, stored as `[2, T]`.

If `hrrp_snr` exists in test files, it is used automatically for testing.

## Installation

```bash
pip install -r requirements.txt
```

## Training

```bash
python code/train_main_angular_domain_corrl_lstm.py \
  --train-dir code/data/train \
  --test-dir code/data/test \
  --metadata code/data/attitute_ground_truth.json \
  --epochs 100 \
  --batch-size 80
```

The best checkpoint is written to `code/outputs/best_cg_lstm.pt` by default.

## Main Files

- `code/train_main_angular_domain_corrl_lstm.py`: training and evaluation entry.
- `code/models/cg_lstm.py`: GMU, cross-attention, CG-LSTM cell, and loss.
- `code/util/dataloader.py`: dataset indexing and ASC correlation computation.
