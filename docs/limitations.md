# Limitations

## 1. Dataset Limitations

The study relies solely on the HAM10000 dataset, which:
- May not reflect global population diversity
- Contains class imbalance
- Is limited to dermoscopic images

External validation on independent datasets was not performed.

## 2. Synthetic Data Quality

Although GAN-generated samples are visually plausible, they may:
- Contain subtle artifacts
- Lack rare pathological patterns
- Fail to capture full clinical variability

No radiologist-level validation of synthetic images was conducted.

## 3. Evaluation Metrics

The primary evaluation focuses on:
- Macro AUC
- Macro F1
- Accuracy

Additional clinical metrics such as sensitivity at fixed specificity thresholds were not evaluated.

## 4. Frozen Backbone Strategy

The classifier backbone (ResNet50) was frozen during training to:
- Reduce computational complexity
- Improve reproducibility

However, full fine-tuning may yield different performance characteristics.

## 5. No External Clinical Validation

The model has not been:
- Validated in real-world hospital environments
- Tested in prospective studies
- Compared with dermatologist performance

## 6. GAN Scope

The GAN is trained using a single-class strategy for minority categories.
While this simplifies conditioning and improves stability, it may limit:
- Multi-class feature interactions
- Cross-class boundary learning

## Conclusion

While GAN-based augmentation improves minority representation in this controlled setting, further validation across datasets and clinical environments is required before practical deployment.