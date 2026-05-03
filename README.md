# Adversarial Robustness Analysis of Deepfake Detection

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kanchisoni/Adversarial-Robustness-Analysis-of-Deepfake-Detection/blob/main/Adversarial_Robustness_Analysis_of_Deepfake_Detection.ipynb)
![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![License](https://img.shields.io/badge/License-MIT-green)

> **Key Finding:** Wavelet-domain preprocessing recovers **47 percentage points**
> of accuracy lost to FGSM adversarial attack — without any model retraining.

---

## What This Project Does

Deepfake detection models trained on clean data achieve near-perfect accuracy — but collapse under even small adversarial perturbations. This project asks:

**Can we defend a deepfake detector against adversarial attacks without retraining it?**

We fine-tune an **Xception model** on 60K real/deepfake images, attack it with **FGSM**, then apply two preprocessing-based defenses — Gaussian blur and Haar wavelet transform — and measure how much accuracy each defense recovers.

---

## Results at a Glance

| Condition | Accuracy |
|---|---|
| Clean validation images | **99%** |
| After FGSM attack (ε = 0.01) | **32%** |
| FGSM + Gaussian Blur defense | **61%** |
| FGSM + Wavelet Transform defense | **79%** |

> Wavelet transform defense recovers **47pp** of lost accuracy. Gaussian blur recovers 29pp.
> Both applied at inference time only — zero retraining required.

---

## Dataset

**Deepfake vs Real — 60,000 Images**

| Property | Detail |
|---|---|
| Source | [Kaggle — prithivsakthiur/deepfake-vs-real-60k](https://www.kaggle.com/datasets/prithivsakthiur/deepfake-vs-real-60k) |
| Size | 60,000 images (Real + AI-generated deepfake faces) |
| Input resolution | 299 × 299 px |
| Train / Val split | 80% / 20% |

---

## Methodology

### 1 — Baseline Model

- **Architecture:** Xception pretrained on ImageNet, fully fine-tuned
- **Task:** Binary classification — Real vs Deepfake
- **Framework:** TensorFlow / Keras
- **Result:** ~99% validation accuracy on clean data

### 2 — Adversarial Attack: FGSM

Fast Gradient Sign Method implemented from scratch using `tf.GradientTape`:

```python
def fgsm_attack(images, labels, model, epsilon=0.01):
    images = tf.cast(images, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(images)
        preds = model(images)
        loss = loss_object(labels, preds)
    gradient = tape.gradient(loss, images)
    signed_grad = tf.sign(gradient)
    adv_images = images + epsilon * signed_grad
    return tf.clip_by_value(adv_images, -1, 1)
```

A single forward-backward pass generates imperceptible noise that drops model accuracy from 99% to 32%.

### 3 — Defenses (Inference-Time Only, No Retraining)

**Gaussian Blur**
Spatial smoothing using a 5×5 kernel via OpenCV. Suppresses high-frequency noise in pixel space. Recovers accuracy to ~61%.

**Haar Wavelet Transform**
Decomposes each image into frequency sub-bands using PyWavelets. Discards high-frequency detail coefficients (where adversarial noise concentrates) and reconstructs from low-frequency components only. Recovers accuracy to ~79%.

```python
def wavelet_defense(images):
    processed = []
    for img in images:
        img = img.numpy()
        coeffs = pywt.dwt2(img[:,:,0], 'haar')
        cA, (cH, cV, cD) = coeffs
        rec = pywt.idwt2((cA, (None, None, None)), 'haar')
        rec = np.stack([rec]*3, axis=-1)
        processed.append(rec)
    return tf.convert_to_tensor(processed)
```

---

## Key Insights

- Adversarial perturbations from FGSM concentrate in **high-frequency image components**
- Frequency-domain defenses (wavelets) outperform spatial defenses (blur) significantly
- A **47-percentage-point recovery** is achievable with a single inference-time preprocessing step
- No model architecture changes or retraining are required — making this practical for deployment

---

## Project Structure

```
├── Adversarial_Robustness_Analysis_of_Deepfake_Detection.ipynb   # Full pipeline
└── README.md
```

The notebook contains the complete pipeline:
- Dataset loading via KaggleHub
- Xception model training
- FGSM attack generation
- Gaussian blur defense
- Wavelet transform defense
- Quantitative evaluation of all conditions

---

## How to Run

**Option 1 — Google Colab (recommended)**

Click the Colab badge at the top. No setup required.

**Option 2 — Local**

```bash
git clone https://github.com/kanchisoni/Adversarial-Robustness-Analysis-of-Deepfake-Detection.git
cd Adversarial-Robustness-Analysis-of-Deepfake-Detection
pip install tensorflow opencv-python pywavelets kagglehub
jupyter notebook
```

> You will need a Kaggle API key configured to download the dataset automatically.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| TensorFlow / Keras | Model training and FGSM implementation |
| Xception | Pretrained backbone (ImageNet weights) |
| OpenCV | Gaussian blur defense |
| PyWavelets | Wavelet transform defense |
| KaggleHub | Dataset download |
| NumPy | Array operations |

---

## Future Work

- Epsilon sensitivity analysis: accuracy vs ε across [0.001, 0.005, 0.01, 0.02, 0.05]
- Stronger attacks: PGD (Projected Gradient Descent), Carlini-Wagner (CW)
- Adversarial training as a defense baseline for comparison
- Extension to video-level deepfake robustness
- Detection of adversarial examples as a separate classification task

---

## Author

**Kanchi Soni**
Undergraduate Researcher — AI Security & Computer Vision

[![GitHub](https://img.shields.io/badge/GitHub-kanchisoni-black?logo=github)](https://github.com/kanchisoni)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
