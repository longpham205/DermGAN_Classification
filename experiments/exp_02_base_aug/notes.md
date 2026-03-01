# Experiment 02 — Class-Aware Basic Augmentation

## Objective
Evaluate the impact of traditional class-aware augmentation on class imbalance.

## Augmentation Strategy
- Minority classes receive stronger transformations.
- Augmentation factors:
  - bcc: ×2
  - akiec: ×3
  - vasc: ×4
  - df: ×4

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
- Improved generalization over baseline.
- Moderate increase in minority recall.
- Limited diversity compared to GAN-generated data.

## Conclusion
Traditional augmentation partially mitigates imbalance but remains limited.