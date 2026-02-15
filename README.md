# Classical Machine Learning Baselines for Deepfake Audio Detection on the Fake-or-Real Dataset

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
  <img src="https://img.shields.io/badge/Dataset-Fake--or--Real-orange" />
  <img src="https://img.shields.io/badge/Best%20Accuracy-93.4%25-brightgreen" />
</p>

> **Authors:** Faheem Ahmad, Ajan Ahmed, Masudul Imtiaz
> **Affiliation:** Clarkson University, Potsdam, New York, USA
> **Contact:** fahmad@clarkson.edu

---

## Table of Contents

1. [Abstract](#abstract)
2. [Motivation](#motivation)
3. [Dataset](#dataset)
4. [Feature Extraction Pipeline](#feature-extraction-pipeline)
   - [Prosodic Features](#1-prosodic-features-21-features)
   - [Voice-Quality Features](#2-voice-quality-features)
   - [Spectral Features](#3-spectral-features-11-features)
5. [Statistical Feature Selection](#statistical-feature-selection)
   - [ANOVA F-test](#anova-f-test)
   - [Missing Value Handling](#missing-value-handling)
   - [Feature Imputation](#feature-imputation)
   - [Dropped Features](#dropped-features)
6. [Exploratory Data Analysis](#exploratory-data-analysis)
7. [Modeling Pipeline](#modeling-pipeline)
   - [Data Splitting](#data-splitting)
   - [Preprocessing](#preprocessing)
   - [Classifiers](#classifiers)
   - [Hyperparameter Tuning](#hyperparameter-tuning)
8. [Results](#results)
   - [44.1 kHz Performance](#441-khz-for-2sec-performance)
   - [16 kHz Performance](#16-khz-for-rerec-performance)
   - [McNemar's Pairwise Tests](#mcnemars-pairwise-statistical-tests)
9. [Key Findings](#key-findings)
10. [Visualizations](#visualizations)
11. [Project Structure](#project-structure)
12. [How to Run](#how-to-run)
13. [Requirements](#requirements)
14. [Research Paper](#research-paper)
15. [Citation](#citation)
16. [License](#license)

---

## Abstract

Deep learning has made it easy to synthesize highly realistic human voices, enabling so-called "deepfake audio" that can be exploited for fraud, impersonation, and disinformation. Despite rapid progress on neural detectors, there remains a need for **transparent baselines** that reveal which acoustic cues most reliably separate real from synthetic speech.

This project presents an **interpretable classical machine learning baseline** for deepfake audio detection using the **Fake-or-Real (FoR) dataset**. We extract a rich set of prosodic, voice-quality, and spectral features from two-second speech clips at two sampling rates (44.1 kHz and 16 kHz), perform statistical analysis (ANOVA, correlation heatmaps), and train 7 classical ML models. The best model, an **RBF SVM**, achieves approximately **93.4% test accuracy** and an **EER of ~6.6%**, while simpler linear models reach around 75% accuracy.

---

## Motivation

- Voice-based authentication systems used in **telephony, banking, and virtual assistants** are particularly vulnerable to spoofed audio.
- While deep learning-based detectors achieve high accuracy, they often operate as **black boxes**, limiting interpretability and trust.
- This work focuses on **interpretable** classical ML baselines that:
  - Reveal **which acoustic features** distinguish real from fake speech
  - Serve as **benchmarks** for future deep models
  - Enable **lightweight, deployable** detection systems for real-world settings

---

## Dataset

We use the **[Fake-or-Real (FoR) dataset](https://bil.eecs.yorku.ca/datasets/)** from York University's APTLY Lab.

| Property | Value |
|---|---|
| **Total audio clips** | 31,138 WAV files (~2.5 GB) |
| **Class distribution** | 15,548 real / 15,590 fake (approximately balanced) |
| **Real speech sources** | CMU Arctic, LJSpeech, VoxForge |
| **Fake speech sources** | Modern TTS systems including WaveNet |
| **Clip duration** | 2 seconds (normalized) |
| **Splits** | Predefined training / testing / validation (speaker-disjoint) |

### Two Sampling-Rate Conditions

| Condition | Description | Sample Rate | Use Case |
|---|---|---|---|
| **for-2sec** | Original normalized 2-second clips | 44.1 kHz | High-fidelity audio analysis |
| **for-rerec** | Re-recorded through speakers & microphones | 16 kHz | Simulating real-world telephony conditions |

### Why No Data Balancing?

The dataset is already approximately balanced (~50/50 split), so we intentionally **did not apply** SMOTE, oversampling, or undersampling. Reasons:
1. No severe class imbalance exists
2. Synthetic oversampling (SMOTE) can distort acoustic feature distributions
3. Standard metrics (accuracy, ROC-AUC, EER) remain meaningful without balancing

---

## Feature Extraction Pipeline

We extract **32 numeric features** per audio clip, divided into three categories:

### 1. Prosodic Features (21 features)

Extracted using **librosa** (YIN algorithm for F0) and parallelized with **joblib** across all CPU cores.

| Feature Group | Features | Count | Description |
|---|---|---|---|
| **F0 (Pitch) Statistics** | `f0_mean_v`, `f0_std_v`, `f0_iqr_v`, `f0_cv_v`, `f0_range_v`, `f0_p10_v`, `f0_p90_v` | 7 | Fundamental frequency statistics over voiced frames using YIN algorithm |
| **RMS Energy** | `rms_mean`, `rms_std`, `rms_cv`, `rms_iqr`, `rms_range` | 5 | Loudness and dynamics of the signal |
| **Voice Quality** | `jitter_local`, `shimmer_local` | 2 | Cycle-to-cycle variation in pitch period and amplitude |
| **Timing/Prosody** | `dur_s`, `voice_pct`, `n_voiced_seg_per_s`, `mean_voiced_seg_ms`, `pause_ratio`, `f0_slope_hz_per_s` | 6 | Temporal structure: voiced percentage, pause ratios, intonation slope |
| **HNR** | `hnr_mean` | 1 | Harmonics-to-noise ratio (always NaN - discarded) |

**Audio preprocessing before extraction:**
- `librosa.load(path, sr=target_sr, mono=True)` for loading and resampling
- `librosa.effects.trim(y, top_db=30)` for silence removal
- Voiced frame detection: frames where F0 > 0 AND RMS > 0.5 * median(RMS)

### 2. Voice-Quality Features

- **Jitter (local):** Median absolute F0 cycle-to-cycle difference / median F0. Natural speech has irregular vocal fold vibration; synthetic speech is often too "perfect."
- **Shimmer (local):** Median absolute amplitude cycle-to-cycle difference / median RMS amplitude.
- **HNR mean:** Attempted but failed extraction (all NaN) - dropped from all analyses.

### 3. Spectral Features (11 features)

Extracted using **librosa** with GPU-accelerated variant available via **torchaudio** on Apple M4 MPS.

| Feature | Stats Computed | Description |
|---|---|---|
| **Spectral Centroid** | mean, std, range | "Center of mass" of the spectrum - correlates with perceived brightness |
| **Spectral Bandwidth** | mean, std, range | Frequency spread around the centroid |
| **Spectral Contrast** | mean, std | Peak-to-valley difference across frequency bands |
| **Spectral Rolloff** | mean, std, range | Frequency below which 85% of spectral energy lies |

**STFT parameters:**

| Condition | Window | Hop | n_fft |
|---|---|---|---|
| 44.1 kHz | 25 ms (1103 samples) | 10 ms (441 samples) | 1103 |
| 16 kHz | 1024 samples | 160 samples | 1024 |

---

## Statistical Feature Selection

### ANOVA F-test

We apply **one-way ANOVA** (`scipy.stats.f_oneway`) to test whether each feature's mean differs significantly between real and fake classes.

- **Null hypothesis (H0):** Feature mean is equal across real vs. fake
- **Threshold:** Features with **p-value >= 0.05** are dropped (not statistically significant)
- Features with high F-statistics indicate strong class separability

### Missing Value Handling

Three-stage pipeline:

1. **Inf/NaN replacement:** All `inf` and `-inf` values replaced with `NaN`
2. **All-NaN column removal:** Columns where every value is NaN are dropped (e.g., `hnr_mean`)
3. **Post-ANOVA NaN check:** After dropping non-significant features, any remaining all-NaN columns are removed

### Feature Imputation

- **Method:** `sklearn.impute.SimpleImputer(strategy='median')`
- Each feature's NaN values are replaced with the **median** of that feature's non-NaN values
- **Median was chosen over mean** because it is robust to outliers common in audio features
- Imputer is **fit on training data only**, then applied to both train and test sets (no data leakage)
- After imputation: **StandardScaler** (z-score normalization) applied, also fit on training data only

### Dropped Features

| Condition | Features Used | Dropped Features | Reason |
|---|---|---|---|
| **44.1 kHz** | 29 features | `hnr_mean`, `spec_bandwidth_rng`, `spec_rolloff_rng` | `hnr_mean`: all NaN; others: p >= 0.05 |
| **16 kHz** | 30 features | `hnr_mean`, `shimmer_local` | `hnr_mean`: all NaN; `shimmer_local`: p ~0.91 (not discriminative) |

---

## Exploratory Data Analysis

### F0 (Pitch) Distribution
- Real speech exhibits **wider, more natural F0 variation** with two broad peaks (male and female pitch ranges)
- Synthetic (TTS) speech shows **tighter, more uniform** F0 distributions with less context-dependent intonation

### Spectral Centroid Distribution
- Real recordings have **higher centroid values** (more high-frequency content from sibilants, breath noise, microphone artifacts)
- Synthetic speech sounds **slightly smoother or muffled**, lacking sharp consonant bursts

### Feature Scatter Plots
- 2D scatter plots of spectral centroid vs. bandwidth show **visible class separation**
- Real clips occupy the high-centroid, high-bandwidth region that fake clips rarely reach

### Correlation Heatmaps
- **Pearson correlation** computed separately for real and fake speech
- Blocks of high correlation appear among spectral features (centroid, bandwidth, rolloff) and among energy features
- **Differences between the two heatmaps** reveal how feature relationships shift between genuine and synthetic speech

---

## Modeling Pipeline

### Data Splitting

- **Method:** `GroupShuffleSplit` (80% train / 20% test)
- **Speaker-disjoint:** Speaker identities do not overlap between train and test sets, preventing speaker leakage
- Training set: ~24,913 clips | Test set: ~6,225 clips

### Preprocessing

```
Raw features → Replace inf with NaN → Drop all-NaN columns → ANOVA filter (p<0.05)
→ Median imputation (fit on train) → Z-score scaling (fit on train) → Model training
```

### Classifiers

| # | Model | Key Parameters | Type |
|---|---|---|---|
| 1 | **Logistic Regression** | L2 regularization, max_iter=1000, class_weight='balanced' | Linear |
| 2 | **LDA** | solver='lsqr', shrinkage='auto' | Linear (Gaussian class-conditional) |
| 3 | **QDA** | reg_param=1e-3 | Quadratic (per-class covariance) |
| 4 | **Gaussian Naive Bayes** | Default | Generative (feature independence) |
| 5 | **Linear SVM** | kernel='linear', C=1.0, class_weight='balanced' | Linear |
| 6 | **RBF SVM** | GridSearchCV: C={0.1, 1, 10}, gamma={'scale', 0.01, 0.001} | Non-linear |
| 7 | **GMM** | 3 components/class, full covariance, log-likelihood ratio | Density-based |

### Hyperparameter Tuning

- **RBF SVM:** 3-fold `StratifiedKFold` cross-validation via `GridSearchCV`
- Scoring metric: `roc_auc`
- Best parameters found: **C=10, gamma='scale'** (both conditions)

---

## Results

### 44.1 kHz (for-2sec) Performance

| Model | Train Accuracy | Test Accuracy | ROC-AUC | EER (%) |
|---|---|---|---|---|
| Logistic Regression | 0.751 | 0.753 | 0.828 | 24.7 |
| LDA | 0.747 | 0.753 | 0.819 | 24.7 |
| QDA | 0.768 | 0.771 | 0.854 | 22.9 |
| Gaussian Naive Bayes | 0.698 | 0.696 | 0.764 | 30.4 |
| Linear SVM | 0.754 | 0.755 | 0.826 | 24.5 |
| **RBF SVM** | **0.953** | **0.927** | **0.980** | **7.3** |
| GMM | 0.787 | 0.783 | 0.869 | 21.7 |

### 16 kHz (for-rerec) Performance

| Model | Train Accuracy | Test Accuracy | ROC-AUC | EER (%) |
|---|---|---|---|---|
| Logistic Regression | 0.756 | 0.754 | 0.834 | 24.6 |
| LDA | 0.753 | 0.752 | 0.827 | 24.8 |
| QDA | 0.774 | 0.771 | 0.857 | 22.9 |
| Gaussian Naive Bayes | 0.701 | 0.698 | 0.769 | 30.2 |
| Linear SVM | 0.758 | 0.758 | 0.830 | 24.2 |
| **RBF SVM** | **0.957** | **0.934** | **0.981** | **6.6** |
| GMM | 0.811 | 0.805 | 0.888 | 19.5 |

### McNemar's Pairwise Statistical Tests

- **LR, LDA, Linear SVM:** No significant pairwise differences (p > 0.4) - similar linear decision boundaries
- **QDA and GMM** significantly outperform linear models (p << 0.05) - non-linear boundaries improve detection
- **RBF SVM** is significantly better than **every other model** (p-values effectively zero)
- **Gaussian Naive Bayes** is significantly worse than all other models (feature independence assumption too restrictive)

**Confirmed model ranking:**

```
RBF SVM >> (GMM ~ QDA) > (LR ~ LDA ~ Linear SVM) >> GNB
```

---

## Key Findings

1. **Best overall model:** RBF SVM achieves **93.4% accuracy**, **0.981 AUC**, and **6.6% EER** at 16 kHz
2. **Pitch variability** (`f0_std_v`, `f0_range_v`, `f0_cv_v`) are among the most discriminative features - real speech has richer, context-dependent intonation
3. **Spectral richness** (`spec_centroid_mean`, `spec_bandwidth_mean`) separates real from fake - real recordings have more high-frequency content
4. **Jitter** captures vocal fold irregularity differences; **shimmer** is NOT discriminative (p ~0.91)
5. **Detection remains robust after re-recording** and downsampling to 16 kHz - key cues survive realistic channel distortions
6. **Classical ML with ~30 hand-crafted features** can rival deep learning approaches on this benchmark
7. **Speaker-disjoint evaluation** prevents overfitting to speaker identity

---

## Visualizations

The notebook generates the following plots:

| Plot | Description |
|---|---|
| `f0_dist_*.png` | F0 (pitch) distribution histograms for real vs. fake |
| `centroid_dist_*.png` | Spectral centroid distribution histograms |
| `scatter_*.png` | Spectral centroid vs. bandwidth 2D scatter plots |
| `corr_heatmap_*.png` | Pearson correlation heatmaps (real vs. fake) |
| `roc_*.png` | ROC curves for all 7 models |
| `det_*.png` | Detection Error Trade-off curves |
| `accuracy_bar_*.png` | Train vs. test accuracy bar charts |
| `far_frr_*.png` | FAR/FRR analysis plots |

All plots are generated for both sampling-rate conditions (`44k_25ms` and `16k_fullclip`).

---

## Project Structure

```
Deepfake-Audio-Detection/
|
|-- FakeorRealByFaheemAhmad_Local_executed.ipynb   # Main notebook (with outputs)
|-- README.md                                       # This file
|-- paper/
|   |-- Classical_ML_Baselines_Deepfake_Audio.pdf   # Research paper
|-- LICENSE                                          # MIT License
```

---

## How to Run

### Prerequisites

```bash
pip install numpy pandas librosa soundfile scipy scikit-learn matplotlib seaborn joblib torch torchaudio
```

### Dataset Setup

1. Download the **Fake-or-Real (FoR)** dataset from [https://bil.eecs.yorku.ca/datasets/](https://bil.eecs.yorku.ca/datasets/)
2. Unzip to a local directory (e.g., `~/Downloads/FoR_root/`)
3. Ensure the directory structure is:
   ```
   FoR_root/
   |-- for-2sec/for-2seconds/{training,testing,validation}/{fake,real}/
   |-- for-rerec/for-rerecorded/{training,testing,validation}/{fake,real}/
   ```

### Running the Notebook

```bash
jupyter notebook FakeorRealByFaheemAhmad_Local_executed.ipynb
```

Or run non-interactively:

```bash
jupyter nbconvert --to notebook --execute FakeorRealByFaheemAhmad_Local_executed.ipynb
```

> **Note:** Full execution takes approximately **35 minutes** on an Apple M4 (10-core GPU) with 31,138 audio files. Feature extraction is the most time-intensive step.

### Hardware Used

| Component | Specification |
|---|---|
| **Chip** | Apple M4 |
| **GPU** | 10-core GPU with Metal 4 support |
| **Framework** | PyTorch 2.9.0 with MPS backend |
| **Spectral extraction** | GPU-accelerated via torchaudio on MPS |
| **Prosodic extraction** | CPU-parallelized via joblib |

---

## Requirements

```
python >= 3.10
numpy >= 1.24
pandas >= 2.0
librosa >= 0.10
soundfile >= 0.12
scipy >= 1.11
scikit-learn >= 1.3
matplotlib >= 3.7
seaborn >= 0.12
joblib >= 1.3
torch >= 2.0
torchaudio >= 2.0
```

---

## Research Paper

The full research paper is available in the [`paper/`](paper/) directory:

**"Classical Machine Learning Baselines for Deepfake Audio Detection on the Fake-or-Real Dataset"**

> Faheem Ahmad, Ajan Ahmed, Masudul Imtiaz
> Clarkson University, Potsdam, New York, USA

The paper covers:
- Detailed motivation and literature review
- Complete feature engineering methodology
- Statistical analysis and feature selection rationale
- All model architectures, hyperparameters, and evaluation metrics
- McNemar's pairwise significance tests
- Discussion on what makes speech sound "real" vs. "fake"
- Limitations and future work directions

---

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{ahmad2025deepfake,
  title={Classical Machine Learning Baselines for Deepfake Audio Detection on the Fake-or-Real Dataset},
  author={Ahmad, Faheem and Ahmed, Ajan and Imtiaz, Masudul},
  institution={Clarkson University},
  year={2025}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <i>Built at Clarkson University | Department of Electrical & Computer Engineering</i>
</p>
