# Experiment 01 — Baseline (Real Images Only)

## Objective
Establish a reference performance using only real dermoscopic images without augmentation.

## Experimental Setup
- Dataset: HAM10000
- Split: lesion-level stratified (70/15/15)
- Model: ResNet50 (pretrained, frozen backbone)
- Loss: Weighted Cross-Entropy
- Primary Metric: Macro AUC
- Early stopping based on validation Macro AUC

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
- Strong bias toward majority class (nv).
- Low recall for minority categories.
- Clear impact of dataset imbalance.

## Conclusion
This experiment serves as the baseline reference for evaluating augmentation strategies.