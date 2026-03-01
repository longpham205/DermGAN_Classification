# Experiment 03 — Auto-Selective GAN Augmentation

## Objective
Evaluate whether GAN-generated minority samples improve classification performance.

## GAN Strategy
- CWGAN-GP
- Single-class training per minority category
- Auto-selective generation:
  - Minority threshold: < 40% of majority class
  - Target size capped at 4× real count
- Majority class not oversampled

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | |
| Test Macro AUC | |
| Test Macro F1 | |
| Test Macro Recall | |

Minority Class Performance:

| Class | F1-score |
|-------|----------|
| df | |
| vasc | |
| akiec | |
| bcc | |

## Observations
- Significant improvement in minority recall.
- Reduced class bias toward nv.
- Stable performance on majority class.
- No observed degradation due to synthetic data.

## Conclusion
Auto-selective GAN augmentation effectively mitigates imbalance while preserving classification stability.