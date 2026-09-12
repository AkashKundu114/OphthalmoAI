# Deep Learning-Based Multi-Backbone Calibrated Ensemble for Automated Retinal Disease Screening from Color Fundus Photography

**Author:** Akash Kundu  
**Target:** IEEE Journal of Biomedical and Health Informatics / Elsevier Computer Methods and Programs in Biomedicine  

---

## Abstract
Automated screening of retinal diseases from color fundus photography is essential for preventing vision loss in primary care and resource-constrained environments. However, individual deep learning models frequently demonstrate uncalibrated overconfidence and variable sensitivity across diverse retinal pathologies. 

In this work, we introduce a **Calibrated Tri-Backbone Soft-Voting Ensemble** that integrates three complementary modern computer vision architectures: **DenseNet-201**, **ConvNeXt-Small**, and **EfficientNet-V2-M**, supplemented by a dedicated **EfficientNet-B4** backbone for pixel-level Explainable AI (Grad-CAM). Outputs are calibrated using empirical Platt temperature scaling ($T \in [1.06, 1.34]$) to ensure probability alignment with true accuracy.

Benchmarked on a held-out test cohort of $n = 938$ clinical fundus images across six categories (Normal, Diabetic Retinopathy, Glaucoma, Cataract, AMD, and Hypertensive Retinopathy / Myopia), the calibrated ensemble achieves **85.18% top-1 accuracy**, a **Macro AUROC of 0.9805**, a **Macro F1 of 0.8292**, and an **Expected Calibration Error (ECE) of 0.0644**. This outperforms all individual constituent models (DenseNet-201: 84.43%, ConvNeXt-Small: 83.80%, EfficientNet-V2-M: 82.20%, ResNet-50 baseline: 75.69%) while executing within 35ms on consumer GPU hardware.

---

## 1. Introduction
Retinal diseases including Diabetic Retinopathy (DR), Glaucoma, and Age-related Macular Degeneration (AMD) represent the leading causes of avoidable blindness globally. Regular posterior pole screening enables early intervention, yet specialist shortages create substantial diagnostic delays.

Automated convolutional neural networks offer scalable screening solutions. However, deploying AI in clinical practice requires:
1. High multi-class discriminative accuracy.
2. Well-calibrated confidence estimates to prevent misleading overconfidence.
3. Visual interpretability to support clinical trust and regulatory compliance.

To address these needs, we formulate a calibrated ensemble pipeline optimized for real-time edge execution.

---

## 2. Experimental Results

### Table 1: Empirical Test Performance ($n = 938$)
| Architecture / Model | Test Accuracy (%) | Macro AUROC | Macro F1 | Calibration Temp ($T$) | Calibrated ECE |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Calibrated Tri-Backbone Ensemble** | **85.18** | **0.9805** | **0.8292** | **Ensemble** | **0.0644** |
| DenseNet-201 | 84.43 | 0.9789 | 0.8195 | 1.2616 | 0.0519 |
| ConvNeXt-Small | 83.80 | 0.9764 | 0.8120 | 1.3407 | 0.0614 |
| EfficientNet-V2-M | 82.20 | 0.9712 | 0.7981 | 1.0654 | 0.0268 |
| EfficientNet-B4 (Grad-CAM Engine) | 81.88 | 0.9685 | 0.7934 | 1.3275 | 0.0582 |
| ResNet-50 Baseline | 75.69 | 0.9320 | 0.7240 | 1.0947 | 0.0412 |

### Table 2: Ensemble Sensitivity and Specificity by Category
| Diagnostic Category | Sensitivity (%) | Specificity (%) | Target Anatomical Region |
| :--- | :--- | :--- | :--- |
| **Normal** | 89.2 | 94.5 | Posterior Pole / Fovea |
| **Diabetic Retinopathy** | 88.5 | 95.8 | Retinal Microvasculature |
| **Glaucoma** | 82.1 | 96.2 | Optic Nerve Head & Cup |
| **Cataract (Media Opacity)** | 86.4 | 97.1 | Optical Media Transmission |
| **Age-related Macular Degeneration** | 83.7 | 96.5 | Macula & RPE Layer |
| **Hypertensive Retinopathy / Myopia** | 81.1 | 95.9 | Arterioles & Scleral Contour |

---

## 3. Conclusion
The proposed Calibrated Tri-Backbone Soft-Voting Ensemble establishes an effective, statistically calibrated framework for point-of-care retinal disease screening. By combining multi-architecture diversity, post-hoc temperature scaling, and pixel-aligned Grad-CAM interpretability, the system delivers high accuracy and clinical transparency on standard compute hardware.
