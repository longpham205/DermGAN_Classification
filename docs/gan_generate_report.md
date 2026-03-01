# GAN Module for Selective Data Augmentation in Skin Lesion Classification

---

# 1. Overview

In skin lesion image datasets, **class imbalance** is a common issue. Certain lesion categories are significantly underrepresented compared to dominant classes, which may lead to:

- Classification bias toward majority classes  
- Reduced recall for rare classes  
- Lower reliability in clinical evaluation scenarios  

To address this issue, the system integrates a **Conditional Wasserstein GAN with Gradient Penalty (CWGAN-GP)** module to generate synthetic images for underrepresented classes in a controlled manner.

**Important principle:**

> The GAN is NOT used to replace real data.  
> It is used only to selectively balance class distribution when necessary.

---

# 2. Objectives of the GAN Module

The GAN module is designed with three primary objectives:

1. Automatically detect underrepresented classes  
2. Generate only the necessary number of synthetic samples  
3. Avoid distorting the original data distribution  

The adopted strategy is:

## Auto-Selective GAN Augmentation

This means:

- Not generating images for all classes  
- Not forcing all classes to have equal sample sizes  
- Only augmenting classes that truly require balancing  

---

# 3. Overall GAN System Pipeline

The GAN pipeline operates as follows:

## Step 1: Data Distribution Analysis

- Count the number of real images per class  
- Identify minority classes using a ratio threshold (e.g., < 40% of the largest class)  
- Compute the number of synthetic samples needed based on median size or a limited growth factor  

## Step 2: Train a Separate GAN per Minority Class

- Each rare class is trained with an independent GAN  
- Prevents label leakage  
- Improves training stability  

## Step 3: Synthetic Image Generation

- Generate exactly the required number of images  
- Store them in a separate `synthetic` directory  
- Do not immediately mix with real training data  

## Step 4: Quality Evaluation

- Perform visual inspection  
- Optionally compute quantitative metrics such as FID  

---

# 4. GAN Architecture

## 4.1 GAN Type

The system employs:

> Conditional Wasserstein GAN with Gradient Penalty (CWGAN-GP)

### Reasons for selection:

- More stable than vanilla GAN  
- Reduces mode collapse risk  
- Suitable for small to medium biomedical datasets  
- Academically interpretable and well-supported in literature  

---

## 4.2 Generator

### Input:
- Noise vector **z**
- Class embedding (conditional input)

### Architecture:
- Fully connected projection layer  
- Series of ConvTranspose2d layers  
- Conditional Batch Normalization  

### Output:
- RGB image  
- Resolution: 128 × 128  
- Pixel range: normalized to [-1, 1]  

### Objective:

The Generator learns to produce synthetic images whose distribution approximates the real data distribution of the target class.

---

## 4.3 Discriminator (Critic)

The Discriminator uses a:

> Projection Discriminator architecture

### Characteristics:

- No sigmoid activation  
- No binary cross-entropy (BCE)  
- No Batch Normalization (as required by WGAN-GP)  
- Uses InstanceNorm for stability  
- Conditioning performed via projection of label embeddings into feature space  

### Output:

- A real-valued **Wasserstein score**

---

## 4.4 Loss Functions (`losses.py`)

### Generator Loss:

```
L_G = - E[D(fake)]

```


### Critic Loss:

```
L_D = E[D(fake)] - E[D(real)] + λ · GP

```


Where:

- **GP** is the Gradient Penalty  
- λ is typically set to 10  
- `n_critic = 5` (the critic is updated more frequently than the generator)  

The Gradient Penalty enforces the Lipschitz constraint required by the Wasserstein formulation.

---

# 5. Directory Structure and File Responsibilities

```
gan/
├── datasets.py
├── evaluate_gan.py
├── generate.py
├── models
│ ├── discriminator.py
│ ├── generator.py
│ └── losses.py
└── train_gan.py

```


---

## 5.1 `datasets.py`

### Functionality:

- Load images for a specific class  
- Resize to 128 × 128  
- Normalize to [-1, 1]  
- Return a DataLoader for GAN training  

### Role:

Provides real training data for minority classes.

---

## 5.2 `models/generator.py`

### Functionality:

- Defines the Generator architecture  
- Integrates conditional input  
- Initializes weights  

### Role:

Generates synthetic images from noise and class labels.

---

## 5.3 `models/discriminator.py`

### Functionality:

- Defines the Projection Discriminator  
- Computes Wasserstein score  
- Applies label conditioning via embedding projection  

### Role:

Distinguishes real and synthetic samples using Wasserstein distance.

---

## 5.4 `models/losses.py`

### Functionality:

- Defines Generator and Critic losses  
- Computes Gradient Penalty  
- Ensures compliance with WGAN-GP formulation  

### Role:

Stabilizes training and enforces Lipschitz constraint.

---

## 5.5 `train_gan.py`

### Functionality:

- Instantiate Generator and Critic  
- Configure optimizers  
- Execute training loop  
- Save checkpoints  
- Log training losses  

### Role:

Trains a dedicated GAN for each minority class.

---

## 5.6 `generate.py`

### Functionality:

- Load trained GAN models  
- Generate required number of images  
- Save synthetic images to disk  

### Role:

Produces synthetic data used for balancing the dataset.

---

## 5.7 `evaluate_gan.py`

### Functionality:

- Compare real vs synthetic images  
- Compute evaluation metrics (e.g., FID)  
- Store analysis results  

### Role:

Ensures synthetic images meet quality standards before integration into the classification pipeline.

---

# 6. Achieved Results

After applying the GAN module:

- Minority classes increased in sample size  
- Class distribution imbalance reduced  
- Recall and F1-score of rare classes improved  
- No degradation in performance of majority classes  

**Important:**

> The GAN does not alter the original data structure.  
> It only adds controlled synthetic samples.

---

# 7. Advantages of the Proposed Method

- Adaptive to dataset distribution  
- Avoids over-generation  
- Does not mechanically force equal class sizes  
- Reproducible  
- Academically well-founded  

---

# 8. Limitations

- Requires separate training for each minority class  
- Synthetic image quality depends on dataset size  
- May require quantitative evaluation metrics such as FID  

---

# 9. Conclusion

The GAN module is designed as a supportive component to:

- Mitigate class imbalance  
- Improve classifier learning capability  
- Preserve the integrity of the original data distribution  

The **Auto-Selective Augmentation Strategy** ensures that the system remains:

- Flexible  
- Stable  
- Academically sound  
