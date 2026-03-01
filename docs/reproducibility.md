# Reproducibility Statement

## Experimental Control

All experiments were conducted under controlled and reproducible settings:

- Fixed random seed: 42
- Deterministic training enabled
- Fixed train/validation/test splits
- Configuration snapshot stored for each experiment

## Configuration Management

Each experiment folder contains:

- `config_snapshot.yaml` — full training configuration
- `notes.md` — experimental observations and results

All hyperparameters, augmentation settings, and model configurations are version-controlled.

## Data Splitting Protocol

A lesion-level stratified split strategy was used to prevent data leakage.
Images from the same lesion do not appear across multiple splits.

Split configuration is stored under:
```
dataset/splits/split_v1.yaml
```


## Logging and Tracking

For every run, the following artifacts are saved:

- Training logs (CSV + TXT)
- Metrics (JSON)
- Classification report
- Confusion matrix
- Predictions (CSV)
- Best model checkpoint

Auto-versioning ensures:

```
results/exp_name/run_001/
results/exp_name/run_002/
```


## GAN Reproducibility

GAN training includes:
- Fixed latent dimension
- Fixed training seed
- Logged FID scores
- Saved generator checkpoints

Auto-selective generation logic is deterministic given identical dataset counts and seed.

## Hardware Environment

Experiments were conducted using GPU acceleration.
Exact hardware configuration may affect training speed but not final deterministic results.

## Code Availability

All configuration files and training scripts are structured to allow full experiment replication.

## Conclusion

The study is designed to ensure strict reproducibility, fair comparison across experiments, and transparent reporting of augmentation strategies.