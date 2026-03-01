# Analysis Module Documentation

---

# 1. Overview of the Analysis Module

The `analysis/` module is designed as a **research-grade evaluation framework** for systematic post-training model assessment.

Its objectives include:

- Evaluating overall model performance  
- Analyzing per-class performance  
- Comparing multiple experiments  
- Performing statistical significance testing  
- Analyzing representation space (feature embeddings)  
- (Optional) Conducting clinical-level interpretability analysis  

The goal extends beyond reporting accuracy. This module establishes a **comprehensive evaluation pipeline** to support:

- Baseline vs. augmentation comparison  
- GAN augmentation assessment  
- Statistical validation of improvements  
- Preparation of publication-ready results  

This design ensures that the system evaluation meets standards expected in medical AI research.

---

# 2. Overall Architecture of the Analysis Pipeline

## Directory Structure

```
src/
└── analysis
├── clinical
├── core
├── performance
├── representation
└── statistics
```

## General Pipeline Flow

```
Experiment Outputs
(metrics.json, predictions.csv, embeddings.npy)
↓
load_results.py
↓
Performance Analysis
↓
Statistical Testing
↓
(Optionally) Representation Analysis
↓
(Optionally) Clinical / Demographic Analysis
```


The pipeline is orchestrated through `run_analysis.py`, which serves as the main entry point.

---

# 3. Core Layer – Foundation of the System

Directory: `analysis/core/`

## 3.1 load_results.py

### Purpose

- Load all relevant experiment outputs
- Standardize input data format for downstream modules

### Main Output: `ExperimentResult`

Contains:

- `exp_name`
- `metrics`
- `classification_report`
- `confusion_matrix`
- `predictions_path`
- `exp_dir`

### Role

This module separates:

- File system operations  
- Analysis logic  

It provides a critical abstraction layer that keeps the pipeline clean, modular, and extensible.

---

## 3.2 compare_experiments.py

### Purpose

Compare multiple experiments using consistent metrics.

### Features

Aggregates:

- Accuracy  
- Macro F1  
- Weighted F1  
- Macro AUC  

Exports:

- Comparison CSV table

### Importance

Enables:

- Baseline vs. augmentation comparison  
- Quantifying GAN impact  
- Generating publication-ready result tables  

---

## 3.3 metrics_utils.py

### Role

- Utility functions for metric computation  
- Ensures consistent metric definitions across modules  

This guarantees reproducibility and metric consistency.

---

# 4. Performance Analysis Layer

Directory: `analysis/performance/`

This layer evaluates model behavior at the prediction level.

---

## 4.1 overall_analysis.py

### Evaluates:

- Accuracy  
- Macro F1  
- AUC  
- Balanced accuracy  

### Purpose

Provides a global assessment of model performance across the dataset.

---

## 4.2 per_class_analysis.py

### Evaluates:

- Precision per class  
- Recall per class  
- F1-score per class  

### Importance

Critical for HAM10000 due to:

- Severe class imbalance  
- Melanoma being a minority class  

Helps determine:

- Whether melanoma is under-detected  
- Whether the model is biased toward majority classes  

---

## 4.3 confusion_analysis.py

### Evaluates:

- Confusion matrix  
- Misclassification patterns  

### Clinical Importance

For example:

- Is melanoma confused with nevus?  
- Is BCC confused with AKIEC?  

Such confusion patterns are highly relevant in medical applications.

---

## 4.4 calibration_analysis.py

### Evaluates:

- Calibration curves  
- Expected Calibration Error (ECE)  

### Importance in Medical AI

Probability estimates must be trustworthy.

A model that is highly confident yet wrong poses clinical risks.

Calibration analysis ensures probabilistic reliability.

---

# 5. Statistics Layer – Statistical Significance Testing

Directory: `analysis/statistics/`

This layer elevates the pipeline to publication standards.

---

## 5.1 bootstrap_ci.py

### Purpose

Compute confidence intervals for:

- Accuracy  
- Macro F1  
- Macro AUC  

### Method

Bootstrap resampling:

- Sampling with replacement  
- N iterations (default: 1000)  
- Percentile-based 95% confidence interval (2.5% – 97.5%)

### Importance

Instead of reporting:

```
Accuracy = 0.87
```

We report:

```
Accuracy = 0.87 (95% CI: 0.84 – 0.90)
```

This aligns with publication standards.

---

## 5.2 mcnemar_test.py

### Purpose

Compare two models evaluated on the same test set.

### Suitable for:

- Baseline vs. augmented models  
- ResNet vs. EfficientNet comparison  

### Hypothesis

Null hypothesis:  
Both models have equivalent performance.

If `p-value < 0.05`, the improvement is statistically significant.

---

## 5.3 significance_report.py

### Aggregates:

- McNemar test  
- Bootstrap confidence intervals  
- Metric comparisons  

### Output

- Statistical summary reports  
- JSON + text report  

### Role

This module transforms the evaluation into a **research-grade statistical framework**.

---

# 6. Representation Analysis Layer

Directory: `analysis/representation/`

---

## 6.1 embedding_analysis.py

### Analyzes:

- Intra-class distance  
- Inter-class distance  
- Compactness  
- Separability  

### Purpose

Evaluate whether GAN augmentation:

- Improves class separation  
- Produces clearer feature clusters  

This operates at the representation level, beyond raw predictions.

---

## 6.2 feature_space_analysis.py

### Analyzes:

- Structure of feature space  
- Clustering behavior  
- Class overlap  

This provides deeper insight into learned representations.

---

# 7. Clinical Layer

Directory: `analysis/clinical/`

---

## 7.1 abcd_analysis.py

### Purpose

Analyze model errors according to the ABCD rule:

- Asymmetry  
- Border  
- Color  
- Diameter  

### Important Note

The HAM10000 dataset does **not** include ABCD annotations.

Therefore:

- This module exists in the pipeline  
- It is not directly applicable to default HAM10000  

To use it, one must:

- Provide a dataset with ABCD annotations  
- Or extract ABCD features via segmentation  

---

# 8. Summary of Module Responsibilities

| Layer | Objective |
|--------|-----------|
| core | Load and standardize experiment data |
| performance | Evaluate prediction-level behavior |
| statistics | Test statistical significance |
| representation | Analyze feature space |
| clinical | Analyze clinical characteristics |

---

# 9. Outputs Generated by the System

The pipeline produces:

- `metrics.json`  
- `experiment_comparison.csv`  
- `confusion_matrix.json`  
- Bootstrap CI reports  
- McNemar test results  
- `embedding_analysis.json`  
- (Optional) clinical reports  

These outputs are sufficient to:

- Construct publication tables  
- Write statistical validation sections  
- Demonstrate GAN augmentation effectiveness  

---

# 10. Scientific Significance of the System

This framework enables:

- Systematic multi-experiment comparison  
- Statistical validation of improvements  
- Deep feature space analysis  
- Reproducibility  
- Publication-ready output generation  

It goes far beyond simply printing accuracy.

---

# 11. Conclusion of the Analysis Module

The Analysis module in the DermGAN system:

- Is modular and well-structured  
- Clearly separates functional layers  
- Minimizes inter-module dependency  
- Is extensible  
- Meets scientific research standards  

For HAM10000, the most critical components are:

- Overall performance analysis  
- Per-class analysis  
- Confusion matrix  
- Bootstrap confidence intervals  
- McNemar significance testing  
- Representation analysis  

The ABCD module is not directly applicable due to missing dataset annotations.

---

This Analysis framework transforms the system from:

> "A model that reports accuracy"

into:

> "A rigorously evaluated, statistically validated, research-grade medical AI system."