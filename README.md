# Intelligent White Blood Cell Classification

An end-to-end deep learning pipeline for White Blood Cell (WBC) classification using advanced preprocessing, context-aware ROI extraction, class imbalance handling, transfer learning, explainable AI, and confidence calibration.

---

## Project Overview

White Blood Cell (WBC) analysis plays a crucial role in diagnosing leukemia, infections, immune disorders, blood cancers, and other hematological diseases. Manual microscopic examination requires significant expertise and is time-consuming. The goal of this project is to automate WBC classification using deep learning while addressing real-world challenges such as class imbalance, noisy microscopy images, morphological similarity between cell types, and model interpretability.

This project evolved through multiple stages of experimentation, beginning with a simpler clinical grouping approach and later progressing to a more challenging 13-class WBC classification system.

---

## Project Journey

### Stage 1: 3-Class Classification

Initially, the dataset was grouped into three clinically meaningful categories:

| Group | Included Classes |
|---------|---------|
| Mature | SNE, LY, MO, BL, EO, BA, BNE, VLY |
| Immature | MY, MMY, PMY |
| Rare | PC, PLY |

### Why?

Instead of directly solving a highly imbalanced 13-class problem, the objective was to:

- Understand dataset behavior
- Analyze class imbalance effects
- Build initial baselines
- Study feature learning behavior
- Identify major challenges

After experimentation, it became clear that grouping multiple biologically different cells into broad categories resulted in loss of important morphological information. Therefore, the project was extended to full 13-class classification.

---

## Stage 2: Full 13-Class Classification

### White Blood Cell Classes

| Class | Cell Type |
|---------|---------|
| SNE | Segmented Neutrophil |
| LY | Lymphocyte |
| MO | Monocyte |
| BL | Blast |
| EO | Eosinophil |
| BA | Basophil |
| BNE | Band Neutrophil |
| VLY | Variant Lymphocyte |
| MY | Myelocyte |
| MMY | Metamyelocyte |
| PMY | Promyelocyte |
| PC | Plasma Cell |
| PLY | Prolymphocyte |

---

# Dataset

## Training Dataset

Merged:

- Phase-1 Training Dataset
- Phase-2 Training Dataset

### Why Merge Both Datasets?

Phase-1 contains relatively clean images while Phase-2 contains more realistic and challenging microscopy images.

Combining both datasets helps the model learn:

- Clean morphology patterns
- Real-world image variability
- Noise robustness
- Better generalization

---

## Test Dataset

Predictions were generated on the Phase-2 Test Dataset.

---

# Class Distribution Analysis

One of the biggest challenges in this project was severe class imbalance.

| Class | Samples |
|---------|---------:|
| SNE | 17,354 |
| LY | 10,801 |
| MO | 3,661 |
| BL | 2,683 |
| EO | 1,148 |
| MY | 588 |
| BA | 554 |
| BNE | 522 |
| VLY | 488 |
| MMY | 480 |
| PMY | 152 |
| PC | 90 |
| PLY | 14 |

### Key Observation

The largest class contains more than 17,000 samples while the smallest class contains only 14 samples.

Without proper imbalance handling, the model would learn majority classes well but completely ignore minority classes.

---

# Exploratory Data Analysis (EDA)

Performed:

- Class distribution analysis
- Sample visualization
- Rare class inspection
- Morphology comparison
- Dataset quality assessment
- Noise analysis

### Findings

- Severe class imbalance exists.
- Multiple classes share similar morphology.
- Some classes are extremely rare.
- Phase-2 images contain substantial noise and artifacts.
- Background RBCs often dominate image regions.

---

# Model Experiments

Several architectures were explored before selecting the final model.

---

## 1. Custom CNN

### Why?

To establish a baseline.

### Observation

- Learned majority classes quickly.
- Performed poorly on minority classes.
- High bias toward common classes.

### Limitation

Could not learn complex morphology effectively.

---

## 2. ResNet

### Why?

Transfer learning provides better feature extraction than training from scratch.

### Observation

- Better generalization
- Improved feature learning
- Stronger performance than CNN

---

## 3. DenseNet121

### Why?

Dense connections improve information flow and feature reuse.

### Observation

- Better learning of cellular structures
- Improved minority-class performance

---

## 4. ConvNeXt-Tiny (Final Model)

### Why ConvNeXt?

ConvNeXt combines the strengths of modern CNN design with strong transfer learning capabilities.

Advantages:

- Better feature representation
- Strong transfer learning performance
- Improved robustness
- Better generalization

Among all tested architectures, ConvNeXt-Tiny provided the best overall balance between accuracy, Macro-F1, and minority-class performance.

---

# Image Preprocessing Pipeline

Microscopy images often contain noise, staining variations, and acquisition artifacts.

To address this, a multi-stage preprocessing pipeline was developed.

---

## Step 1: Denoising

### Techniques Used

1. Gaussian Blur
2. Bilateral Filtering
3. CLAHE (Contrast Limited Adaptive Histogram Equalization)

### Why?

The objective was to:

- Reduce image noise
- Improve local contrast
- Preserve important cell structures
- Improve ROI extraction quality

### Observation

Denoising improved image quality for many samples but was less effective for heavily degraded images.

---

# Context-Aware ROI Cropping

One of the most important contributions of this project.

---

## Why ROI Cropping?

Original images contain:

- Large background regions
- Numerous RBCs
- Empty space
- Imaging artifacts

These regions do not contribute significantly to WBC classification.

---

## Initial Idea

Crop only the nucleus.

### Problem

Important biological information was lost.

Many classes depend on:

- Cytoplasm
- Granules
- Chromatin patterns
- Cell boundaries
- Overall cell morphology

Nucleus-only cropping removed these features.

---

## Context-Aware ROI Solution

Instead of cropping only the nucleus, the pipeline detects the WBC and preserves surrounding cellular context.

Detection methods included:

- HSV-based detection
- LAB color-space analysis
- Saturation masking
- Center-crop fallback

Additional padding was added around the detected cell.

---

## Why Context Matters

Different classes rely on different visual cues.

| Class | Important Feature |
|---------|---------|
| SNE | Multi-lobed nucleus |
| BNE | Band-shaped nucleus |
| EO | Orange granules |
| BA | Dark purple granules |
| LY | Large nucleus, small cytoplasm |
| VLY | Irregular lymphocyte morphology |
| MO | Kidney-shaped nucleus |
| BL | Blast chromatin pattern |
| MY | Cytoplasmic granules |
| MMY | Indented nucleus |
| PMY | Prominent nucleoli |
| PC | Clock-face nucleus |
| PLY | Prolymphocyte morphology |

Removing context would eliminate many of these diagnostic features.

---

# Data Augmentation

### Why?

To improve generalization and increase minority-class representation.

### Techniques

- Horizontal Flip
- Vertical Flip
- Rotation
- Brightness Adjustment
- Contrast Adjustment

---

## Rare-Class Augmentation

Special augmentation was applied to:

- PMY
- PC
- PLY

### Goal

Increase representation of rare classes and improve minority-class learning.

---

# Class Imbalance Handling

Since class imbalance was the biggest challenge, multiple techniques were combined.

---

## WeightedRandomSampler

### Why?

Ensures minority classes appear more frequently during training.

### Benefit

Reduces majority-class dominance.

---

## Class Weights

### Why?

Errors on rare classes should be penalized more heavily.

### Benefit

Encourages learning of minority classes.

---

## Hybrid Loss

### Components

- Cross Entropy Loss
- Focal Loss

### Why?

Cross Entropy provides stable learning.

Focal Loss focuses on difficult and minority-class samples.

### Benefit

Improved Macro-F1 and minority-class recognition.

---

# Transfer Learning Strategy

The final model used ConvNeXt-Tiny pretrained on ImageNet.

---

## Stage 1: Frozen Backbone

Only classifier layers were trained.

### Why?

Preserve pretrained knowledge while adapting to WBC images.

---

## Stage 2: Fine-Tuning

Entire network was trained.

### Why?

Allow deeper adaptation to cellular morphology.

---

# Training Strategy

## 3-Fold Cross Validation

### Why?

A single train-validation split may produce misleading results.

Cross-validation provides:

- More reliable evaluation
- Better generalization estimates
- Reduced variance

---

## Additional Techniques

- Learning Rate Scheduling
- Early Stopping
- Checkpoint Saving
- Best Model Selection

---

# GradCAM Analysis

After training, GradCAM was used to visualize model attention.

---

## Why GradCAM?

To answer important questions:

- Is the model focusing on the nucleus?
- Is the model focusing on granules?
- Is the model using biologically meaningful regions?
- Is the model relying on background shortcuts?

---

## Observation

GradCAM showed that successful predictions were generally based on:

- Nucleus morphology
- Cytoplasmic structures
- Granule patterns
- Chromatin texture

This increased confidence that the model was learning meaningful biological features.

---

# Error Analysis

Detailed confusion matrix analysis was performed.

---

## Most Challenging Classes

| Class | Main Reason |
|---------|---------|
| BNE | Similar to SNE |
| VLY | Similar to LY |
| MY | Similar to MMY |
| PMY | Similar to BL |
| PLY | Extremely limited samples |

---

## Key Insight

Most errors occurred between morphologically similar cell maturation stages.

This indicates that the challenge is biological similarity rather than random prediction errors.

---

# Confidence Calibration

Neural networks often produce overconfident probabilities.

To improve reliability, calibration techniques were applied.

---

## Temperature Scaling

### Why?

Calibrate confidence scores without changing model predictions.

### Benefit

More reliable probability estimates.

---

## Threshold Calibration

Per-class threshold optimization was performed.

### Why?

Different classes require different confidence thresholds.

### Benefit

- Improved minority-class recognition
- Improved Macro-F1
- Better balanced predictions

---

# Validation Results

## Before Calibration

| Metric | Score |
|---------|---------|
| Validation Macro-F1 | 0.6797 |

---

## After Calibration

| Metric | Score |
|---------|---------|
| Validation Accuracy | 89.66% |
| Validation Macro-F1 | 0.7189 |
| Validation Balanced Accuracy | 0.7570 |

---

# Cross-Validation Results

| Metric | Score |
|---------|---------|
| Accuracy | 0.8130 |
| Balanced Accuracy | 0.7775 |
| Macro Precision | 0.7239 |
| Macro Recall | 0.7775 |
| Macro F1 | 0.7366 |
| Weighted F1 | 0.8303 |

---

# Final Pipeline

Raw Images

↓  

Exploratory Data Analysis

↓

Denoising
(Gaussian + Bilateral + CLAHE)

↓

Context-Aware ROI Cropping

↓

Resize (224×224)

↓

Data Augmentation

↓

Class Imbalance Handling

↓

ConvNeXt-Tiny Transfer Learning

↓

3-Fold Cross Validation

↓

GradCAM Analysis

↓

Temperature Scaling

↓

Threshold Calibration

↓

Final Predictions

---

# Key Contributions

- Developed an end-to-end WBC classification pipeline.
- Explored both 3-class and 13-class classification strategies.
- Compared CNN, ResNet, DenseNet121, and ConvNeXt-Tiny architectures.
- Designed a context-aware ROI extraction pipeline.
- Implemented advanced denoising techniques.
- Addressed severe class imbalance using augmentation, WeightedRandomSampler, class weights, and Hybrid Loss.
- Applied transfer learning and fine-tuning.
- Performed 3-Fold Cross Validation.
- Conducted GradCAM-based explainability analysis.
- Performed detailed error analysis.
- Applied temperature scaling and threshold calibration to improve minority-class performance.

---

# Technologies Used

- Python
- PyTorch
- OpenCV
- NumPy
- Pandas
- Scikit-Learn
- timm
- Matplotlib
- Seaborn
- GradCAM

---

# Future Improvements

- ConvNeXt V2
- Vision Transformers (ViT)
- Swin Transformers
- Foundation Models for Medical Imaging
- Ensemble Learning
- Multi-Scale Attention Mechanisms
- Self-Supervised Pretraining

---

# Author

Goundla Lasya Priya

B.Tech Computer Science and Engineering (AI & ML)

Areas of Interest:

- Artificial Intelligence
- Machine Learning
- Deep Learning
- Computer Vision
- Medical Image Analysis
- Healthcare AI
