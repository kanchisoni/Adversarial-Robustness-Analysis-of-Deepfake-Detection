# Adversarial-Robustness-Analysis-of-Deepfake-Detection
This project analyzes the adversarial robustness of a CNN-based deepfake detection system. FGSM attacks are used to generate adversarial samples, causing severe accuracy drops. Frequency-domain and preprocessing defenses such as Gaussian blur and wavelet transforms are applied to improve robustness without retraining the model.
Nice choice — a **strong README** makes this project look *research-grade*.
Below is a **complete `README.md`** you can directly paste into GitHub.
I’ve added **image placeholders + graph sections** (you just replace paths with your actual outputs).

---

 Adversarial Robustness Analysis of Deepfake Detection

![Image](https://media.springernature.com/lw1200/springer-static/image/art%3A10.1007%2Fs00521-024-10181-7/MediaObjects/521_2024_10181_Fig1_HTML.png)

![Image](https://www.researchgate.net/publication/336402462/figure/fig1/AS%3A812471887609871%401570719801771/An-adversarial-example-generated-by-the-FGSM-attack-16-on-the-VGG-16-network-55.jpg)

![Image](https://images.openai.com/static-rsc-3/q6uJrz29SDbxyfDgLA1CM0wt-tJAaSM7IqQya-XqX9nlZsYPYrOMLnEZa3WxPRcEM6KNS-OFKjfq2KJs4FqXJt0PoE2f5obkUZ6QbsRhCls?purpose=fullsize\&v=1)

![Image](https://miro.medium.com/1%2ArYdDFyAiUekH_iQRfMTpYA.png)

📌 Overview

Deepfake detection models based on deep learning achieve high accuracy on clean data but are highly vulnerable to **adversarial attacks**.
This project analyzes the robustness of a CNN-based deepfake detector under **FGSM adversarial perturbations** and evaluates simple **preprocessing and frequency-domain defenses** to improve resilience without retraining the model.

🎯 Objectives

* Evaluate vulnerability of deepfake detectors to adversarial attacks
* Generate adversarial samples using FGSM
* Apply preprocessing and frequency-domain defenses
* Compare robustness improvements using quantitative metrics

🧠 Methodology

1️⃣ Baseline Deepfake Detection

* CNN trained for **binary classification** (Real vs Deepfake)
* Achieves high accuracy on clean validation images

2️⃣ Adversarial Attack: FGSM

* Uses gradient-based perturbations
* Small noise added to input images
* Causes significant accuracy degradation

![Image](https://www.researchgate.net/publication/336402462/figure/fig1/AS%3A812471887609871%401570719801771/An-adversarial-example-generated-by-the-FGSM-attack-16-on-the-VGG-16-network-55.jpg)

![Image](https://www.tensorflow.org/static/tutorials/generative/images/adversarial_example.png)

3️⃣ Defensive Preprocessing

Applied **without retraining or architecture changes**:

* Gaussian Blur
* Wavelet Transform (frequency-domain)
* Feature-space denoising

These techniques aim to suppress **high-frequency adversarial noise**.

![Image](https://upload.wikimedia.org/wikipedia/commons/e/e0/Jpeg2000_2-level_wavelet_transform-lichtenstein.png)

![Image](https://images.openai.com/static-rsc-3/FqZSyCEzOfDXR2h7e8TprXlqxhhsVpS5cHZ11Qu7vx4IM7oJSuUha964R-kcf46v2Hdka-3bJSqDIls0q55mIJqvgFLYLFjTZu3x3VlKZWU?purpose=fullsize\&v=1)

4️⃣ Evaluation Strategy

* Clean validation images
* FGSM adversarial images
* Adversarial images + preprocessing defenses

📊 Experimental Results
 Accuracy Comparison

| Scenario                 | Accuracy |
| ------------------------ | -------- |
| Clean validation images  | **99%**  |
| FGSM adversarial images  | **32%**  |
| FGSM + Gaussian Blur     | **61%**  |
| FGSM + Wavelet Transform | **79%**  |

📈 **Observation:** Frequency-domain defenses significantly outperform spatial smoothing.

📉 Graphical Analysis

🔹 Accuracy vs Defense Method

*(Insert your generated plot here)*

```
results/accuracy_comparison.png
```

![Image](https://www.researchgate.net/profile/Md-Mashfiquer-Rahman/publication/394844154/figure/fig2/AS%3A11431281601317759%401755864816454/Bar-chart-comparing-the-performance-of-the-AI-model-under-different-conditions_Q320.jpg)

![Image](https://images.prismic.io/encord/24759076-3d94-4e2a-a540-d5ed93546256_Robustness%2Bvs%2BAccuracy%2B-%2BEncord.png?auto=compress%2Cformat)

🔹 Effect of FGSM Noise

*(Insert visualization of perturbations)*

```
results/fgsm_noise_visualization.png
```

🔍 Key Insights

* Adversarial noise mainly resides in **high-frequency components**
* CNN feature representations become more stable after frequency transforms
* Simple preprocessing defenses can greatly enhance robustness
* No retraining or model modification is required

🧪 Technologies Used

* Python
* TensorFlow / PyTorch
* OpenCV
* NumPy
* Scikit-learn

📂 Project Structure

```
├── data/
├── model/
│   └── cnn_model.py
├── attacks/
│   └── fgsm.py
├── defenses/
│   ├── gaussian_blur.py
│   └── wavelet_transform.py
├── evaluation/
├── results/
│   ├── accuracy_comparison.png
│   └── fgsm_noise_visualization.png
└── README.md
```

---

🚀 Applications

* Secure deepfake detection systems
* Media forensics and misinformation detection
* Adversarial robustness research
* Trustworthy AI deployment

-📌 Future Work

* Stronger attacks (PGD, CW)
* Feature-space adversarial detection
* Defense-aware training
* Video-level robustness analysis
conference-style README**
* Optimize it for **internship / PhD applications


