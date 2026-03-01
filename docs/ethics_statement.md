# Ethics Statement

## Dataset Usage

This study utilizes the HAM10000 dataset, a publicly available collection of dermoscopic images for research purposes. The dataset contains anonymized images and associated diagnostic labels. No personally identifiable information (PII) is included.

All data usage complies with the original dataset licensing and ethical approval conditions.

## Clinical Scope

The developed classification system is intended for research purposes only and is not designed for clinical deployment. The model does not replace professional medical diagnosis.

Automated predictions should never be interpreted as definitive medical decisions without expert review.

## Bias and Fairness Considerations

Skin lesion datasets may suffer from:
- Demographic imbalance
- Imaging device variation
- Geographic bias

Although class imbalance is addressed through GAN-based augmentation, demographic fairness cannot be fully guaranteed due to limitations in the source dataset.

Future work should incorporate:
- Multi-center data
- Diverse skin tone representation
- Clinical validation studies

## Synthetic Data Responsibility

GAN-generated images are used exclusively for:
- Data augmentation
- Improving minority class representation

Synthetic images are not presented as real patient data.

The generation process follows a minority-targeted strategy to reduce imbalance rather than artificially inflating overall dataset size.

## Risk Assessment

Potential risks include:
- Overfitting to synthetic artifacts
- False confidence in model generalization

Mitigation strategies include:
- Fixed train/val/test splits
- Evaluation on untouched real test data
- Monitoring macro-level performance metrics

## Conclusion

This research adheres to responsible AI principles and aims to improve fairness in imbalanced medical image classification while acknowledging current limitations.