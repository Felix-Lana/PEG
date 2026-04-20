# PEGUNet: Perception-Enhanced Gated U-Net for Robust Multi-Structure Anterior Segment Segmentation in Scheimpflug Imaging

[![Paper](https://img.shields.io/badge/Paper-Knowledge--Based%20Systems-blue)](https://doi.org/10.1016/j.knosys.2026.115976)
[![Python](https://img.shields.io/badge/Python-3.9%2B-green)](#installation)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red)](#installation)

Official implementation of **PEGUNet**, a perception-enhanced gated segmentation network for **four-class joint segmentation** of Scheimpflug anterior segment optical section images.

PEGUNet is designed for robust segmentation of:
- **Background**
- **Cornea**
- **Anterior chamber**
- **Lens**

It specifically addresses two clinically common challenges in real-world Scheimpflug imaging:
1. **Specular highlight interference**
2. **Lens invisibility / weak lens visibility**

The model performs **fully automatic end-to-end inference** and does **not rely on rule-based post-processing** in the main pipeline. An **optional SAM-assisted local editing module** is also provided for rare long-tail failure correction and annotation refinement. :contentReference[oaicite:3]{index=3}

---

## News

- **[Accepted & Published]** Our paper has been accepted by and published in **Knowledge-Based Systems**.
- Code, trained weights, training/inference scripts, and evaluation utilities are released for reproducibility. :contentReference[oaicite:4]{index=4} :contentReference[oaicite:5]{index=5}

---

## Abstract

Rotating Scheimpflug anterior segment imaging is an important tool for quantitative analysis of corneal and anterior chamber structures. However, stable multi-structure segmentation under real clinical conditions remains difficult because of substantial distribution shifts. Specular highlights may weaken boundary evidence, while poor lens visibility or lens absence may disturb structural priors and lead to systematic errors at shared boundaries. To address these issues, we propose **PEGUNet**, a perception-enhanced gated segmentation network built on **UNet 3+** for four-class joint segmentation of Scheimpflug optical section images. PEGUNet introduces a **perception module** to jointly model spatial, frequency-domain, and curvature-guided information, and a **gated multi-scale fusion module** to suppress unreliable responses across scales. On a private clinical dataset of **787 eyes**, PEGUNet achieved a **mean Dice of 0.9782** and a **mean HD95 of 1.5679** on the standard test set, while also showing consistent gains on an independent challenging subset.

---

## Highlights

- **Robust four-class segmentation** for Scheimpflug anterior segment images
- **UNet 3+ backbone** with perception-enhanced feature learning
- **Spatial + frequency + curvature** collaborative representation
- **Gated multi-scale fusion** for improved shared-boundary consistency
- **Optional SAM-assisted interactive refinement** for rare hard cases
- **Full training / inference / evaluation pipeline** for reproducibility 

---

## Method Overview

PEGUNet is built on **UNet 3+** and introduces two key components:

### 1. Perception Module (PM)
The perception module enhances feature discriminability by jointly modeling three complementary cues:

- **Spatial Perception Unit (SPU)**  
  Captures local and multi-scale spatial context and improves sensitivity to thin anatomical boundaries.

- **Frequency Perception Unit (FPU)**  
  Uses frequency-inspired representations to better distinguish true boundaries from artifact-induced high-frequency responses.

- **Curvature Perception Unit (CPU)**  
  Incorporates first- and second-order structural cues to improve continuity and reduce false boundary activation caused by highlights.

These three branches are adaptively fused with attention to produce perception-enhanced features.

### 2. Multi-Scale Gated Fusion
Before full-scale aggregation, PEGUNet applies a gated fusion strategy to suppress noisy or unreliable cross-scale responses. This helps reduce:
- artifact propagation,
- cross-class confusion,
- boundary inconsistency,
- topology errors near shared interfaces. 

---

## Performance

### Standard Test Set
PEGUNet achieved:

- **Average Dice:** 0.9782
- **Average IoU:** 0.9580
- **Average Precision:** 0.9792
- **Average Recall:** 0.9778
- **Average HD95:** 1.5679
- **Average ASSD:** 0.3599

These results outperformed several strong CNN, Transformer, and Mamba-based baselines under a unified training and evaluation protocol.

### Challenging Clinical Cases
In addition to routine testing, the paper evaluates robustness on a held-out challenging subset containing:
- **Highlight-shift subset (H): 61 cases**
- **Lens-invisible subset (M): 41 cases**
- **Total challenging subset (C): 102 cases**

PEGUNet maintains consistent gains under these clinically relevant distribution shifts, showing improved robustness to highlight interference and lens visibility degradation. 

---

## Dataset

The experiments were conducted on a **private clinical dataset of 787 Scheimpflug anterior segment optical section images from 423 subjects**. Images were annotated into four classes:
- background,
- cornea,
- anterior chamber,
- lens.


