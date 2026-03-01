# Preprocessing Pipeline Report  
**Project:** DermGAN_Classification  
**Dataset:** HAM10000  

---

# 1. Overview

## 1.1. Role of Preprocessing in Skin Lesion Classification

The HAM10000 dataset consists of 10,015 dermoscopic images with significant variability:

- Non-uniform image resolutions  
- Different lighting conditions  
- Hair artifacts  
- Dark borders from acquisition devices  
- Irrelevant background noise  

If these images are fed directly into a CNN model:

- Training gradients may become unstable  
- The model may learn noise instead of lesion-specific features  
- It becomes difficult to fairly compare different experiments  

Therefore, preprocessing is a mandatory step to:

- Standardize input data  
- Remove irrelevant noise  
- Improve reproducibility  
- Ensure fair comparison between baseline and GAN-based augmentation experiments  

---

# 2. System Pipeline Design

## 2.1. Design Principles

The preprocessing pipeline follows these principles:

- **Config-driven**: All parameters are defined in `configs/experiment.yaml`
- **Split-first strategy**: Train/validation/test split is performed before preprocessing (lesion-level)
- **No data leakage**
- **Single processed image folder**
- **Reproducible via fixed seed**
- **Modular preprocessing components**

---

## 2.2. Overall Processing Flow

```
RAW DATA (HAM10000)
│
▼
generate_splits.py
(lesion-level stratified split)
│
▼
dataset/splits/
├── train.csv
├── val.csv
└── test.csv
│
▼
run_preprocess.py
│
▼
dataset/processed/images/
├── ISIC_0000001.jpg
├── ISIC_0000002.jpg
└── ...
```


---

# 3. Detailed Preprocessing Pipeline

## 3.1. Processing Steps

The current pipeline includes:

---

### Step 1 — Resize

- Resize all images to a fixed resolution (e.g., 224×224)
- Compatible with CNN architectures such as ResNet and EfficientNet
- Reduces input size variability

**Purpose:**

- Standardize model input
- Reduce computational cost

---

### Step 2 — Pixel Normalization

- Scale pixel values to [0, 1]
- Optionally normalize using ImageNet mean/std for pretrained models

**Purpose:**

- Stabilize gradients
- Accelerate convergence

---

### (Optional) Step 3 — Hair Removal

Hair artifacts are removed using:

- Morphological operations
- Image inpainting

**Purpose:**

- Prevent the model from learning hair-related artifacts
- Improve lesion boundary learning

---

### (Optional) Step 4 — Color Constancy

Normalize illumination differences across images.

**Purpose:**

- Reduce device-related bias
- Improve generalization ability

---

### (Optional) Step 5 — Lesion Cropping

Crop the main lesion region (ROI).

**Purpose:**

- Reduce background noise
- Focus learning on pathological features

---

# 4. Directory Structure

```
preprocessing/
├── resize_normalize.py
├── color_constancy.py
├── hair_removal.py
├── lesion_crop.py
└── split_data.py.py

dataset/
├── splits/
│ ├── train.csv
│ ├── val.csv
│ └── test.csv
├── processed/
│ ├── train/
│ ├── val/
│ └── test/
```


---

# 5. Responsibilities of Each File

## 5.1. generate_splits.py

**Functions:**

- Load metadata
- Perform lesion-level stratified split
- Load label IDs from configuration
- Generate:
  - train.csv
  - val.csv
  - test.csv
  - split_v1.yaml

**Guarantees:**

- No lesion-level data leakage
- 100% reproducibility

---

## 5.2. resize_normalize.py

- Resize images
- Normalize pixel values

This is the minimum required preprocessing step.

---

## 5.3. hair_removal.py (Optional)

- Detect hair artifacts
- Apply inpainting

Helps improve segmentation and classification performance.

---

## 5.4. color_constancy.py (Optional)

- Normalize lighting conditions
- Reduce domain shift

---

## 5.5. lesion_crop.py (Optional)

- Crop lesion ROI
- Remove unnecessary background

---

## 5.6. run_preprocess.py

Entry point of the preprocessing pipeline.

**Responsibilities:**

- Load configuration
- Read train.csv / val.csv / test.csv
- Iterate over images
- Apply preprocessing steps
- Save processed images to:


```
dataset/processed/images/
```


---

# 6. Design Choice: No Separate Folders After Split

Instead of:
```
processed/
├── train/
├── val/
└── test/
```


We use:
```
processed/
```

**Reasons:**

- Split information is stored in CSV files
- No need to duplicate images
- Cleaner directory structure
- PyTorch Dataset reads data directly from CSV

This design is modern, clean, and efficient.

---

# 7. Results After Preprocessing

After running the pipeline:

All images have:

- Uniform resolution
- Normalized pixel values
- Basic noise removal
- No train-test leakage

The dataset remains:

- 10,015 images
- 7 classes

Class distribution and lesion checksum are recorded in `split_v1.yaml`.

---

# 8. Reproducibility

Reproducibility is ensured by:

- Fixed random seed
- Lesion-level stratified split
- Config-driven label mapping
- Lesion checksum stored in split_v1.yaml
- No uncontrolled random shuffling

This enables:

- Fair comparison between baseline and GAN augmentation
- Reporting mean ± standard deviation across multiple runs

---

# 9. Suitability for GAN-Based Research

The preprocessing pipeline is designed to support:

- Baseline experiments using real images
- Future GAN-based synthetic augmentation
- Fixed original split without modification
- Fair and controlled experimental comparison

Therefore, preprocessing serves as the foundation for the entire research framework.

---

# 10. Conclusion

The preprocessing system:

- Standardizes input data
- Prevents data leakage
- Ensures reproducibility
- Is modular and extensible
- Supports both baseline and GAN-based experiments

This design follows:

- Reproducible research standards
- Best practices in machine learning pipelines
- Publication-level experimental rigor