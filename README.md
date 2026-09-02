# Image Dehazing using Multi-Scale Deep Learning

> An AI-based image dehazing system developed for **Smart India Hackathon 2023**, designed to restore visibility in hazy images using multi-scale feature extraction, attention mechanisms, and residual learning.

[GitHub Repository](https://github.com/saturnextreme/final-yea-proj)

---

## Overview

Image dehazing is the process of recovering a clear image from a degraded image affected by atmospheric haze. Haze reduces visibility, contrast, and color information, which can negatively affect computer vision systems and human interpretation.

This project implements a deep learning-based image restoration model using **PyTorch** and the **RESIDE dataset**.

The proposed architecture combines a lightweight **MobileNetV3-Small encoder**, multi-scale feature extraction, **latent multi-head attention**, feature fusion, skip connections, and residual learning to reconstruct clearer images from hazy inputs.

The model is trained and evaluated using image restoration metrics including **PSNR** and **SSIM**.

---

## Key Features

* Multi-scale image feature extraction
* MobileNetV3-Small pretrained backbone
* Latent multi-head attention
* Feature fusion across multiple scales
* Skip connections for improved feature reconstruction
* Residual learning for image restoration
* Smooth L1 reconstruction loss
* EfficientNet-based perceptual loss
* Mixed-precision training when CUDA is available
* Gradient clipping for training stability
* OneCycleLR learning-rate scheduling
* Early stopping
* PSNR and SSIM evaluation
* Reproducible train/validation/test split

---

## Model Architecture

The model is built around a multi-scale encoder-decoder architecture.

```text
                 Hazy Input Image
                        │
                        ▼
             MobileNetV3-Small
                Feature Encoder
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
      Scale 1        Scale 2       Scale 3
      Features       Features      Features
          │             │             │
          └─────────────┼─────────────┘
                        ▼
              Latent Multi-Head
                   Attention
                        │
                        ▼
                Feature Fusion
                        │
                        ▼
              Convolutional Decoder
                        │
                 Skip Connections
                        │
                        ▼
                 Residual Learning
                        │
                        ▼
                Dehazed Output
```

The encoder extracts features at multiple resolutions. These features are processed through latent multi-head attention before being fused and passed to the decoder.

Skip connections preserve useful spatial information from earlier feature extraction stages, while residual learning helps the network reconstruct the clean image from the degraded input.

---

## Dataset

The model is trained using the **RESIDE (REalistic Single Image DEhazing)** dataset.

The dataset is divided into:

* **70% Training**
* **15% Validation**
* **15% Testing**

A fixed random seed is used for reproducibility.

---

## Training Pipeline

The training process consists of the following stages:

```text
Dataset
   │
   ▼
Image Loading & Preprocessing
   │
   ▼
Train / Validation / Test Split
   │
   ▼
Multi-Scale Feature Extraction
   │
   ▼
Latent Multi-Head Attention
   │
   ▼
Feature Fusion
   │
   ▼
Decoder + Skip Connections
   │
   ▼
Residual Reconstruction
   │
   ▼
Loss Calculation
   │
   ├── Smooth L1 Loss
   │
   └── Perceptual Loss
   │
   ▼
Backpropagation
   │
   ▼
OneCycleLR Scheduling
   │
   ▼
Validation
   │
   ▼
Early Stopping
```

---

## Loss Functions

The training objective combines reconstruction and perceptual information.

### Smooth L1 Loss

Smooth L1 loss is used as the primary reconstruction loss to measure the difference between the predicted dehazed image and the target image.

### Perceptual Loss

Perceptual features are extracted using an **EfficientNet-B0** feature network. This provides a higher-level comparison between generated and target images beyond direct pixel-level differences.

The combined loss helps the model optimize both image reconstruction quality and perceptual similarity.

---

## Training Optimizations

Several techniques were implemented to improve the training process:

### Mixed Precision

Mixed-precision training is used when CUDA is available to reduce memory usage and improve training efficiency.

### Gradient Clipping

Gradient clipping is applied to help maintain training stability.

### OneCycleLR

A `OneCycleLR` learning-rate scheduler is used to dynamically adjust the learning rate throughout training.

### Early Stopping

Training can stop when validation performance stops improving, helping reduce unnecessary training and overfitting.

---

## Evaluation

The model is evaluated using two commonly used image restoration metrics.

### PSNR

**Peak Signal-to-Noise Ratio (PSNR)** measures the pixel-level reconstruction quality of the restored image.

Higher PSNR generally indicates better reconstruction quality.

### SSIM

**Structural Similarity Index Measure (SSIM)** evaluates structural similarity between the restored image and the reference image.

Higher SSIM indicates greater structural similarity.

---

## Results

The project evaluates the trained model using both PSNR and SSIM.

The training pipeline is configured to report:

```text
PSNR
SSIM
```

These metrics provide complementary measurements of reconstruction quality.

---

## Technology Stack

| Component            | Technology             |
| -------------------- | ---------------------- |
| Programming Language | Python                 |
| Deep Learning        | PyTorch                |
| Backbone             | MobileNetV3-Small      |
| Perceptual Features  | EfficientNet-B0        |
| Computer Vision      | OpenCV                 |
| Dataset              | RESIDE                 |
| Evaluation           | PSNR, SSIM             |
| Training             | CUDA / Mixed Precision |

---

## Project Structure

```text
final-yea-proj/
│
├── train.py
├── README.md
└── ...
```

The main training implementation is contained in `train.py`, including:

* Dataset loading
* Data splitting
* Model architecture
* Multi-scale feature extraction
* Latent multi-head attention
* Feature fusion
* Decoder
* Loss functions
* Optimizer
* Learning-rate scheduling
* Training loop
* Validation
* Evaluation metrics
* Early stopping

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/saturnextreme/final-yea-proj.git
cd final-yea-proj
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Linux / macOS**

```bash
source venv/bin/activate
```

**Windows**

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

Install the required Python packages used by the project.

```bash
pip install torch torchvision opencv-python numpy pillow
```

Additional dependencies may be required depending on the exact environment and dataset configuration.

### 4. Prepare the Dataset

Download and prepare the required **RESIDE** dataset according to the dataset organization expected by the training script.

### 5. Start Training

Run:

```bash
python train.py
```

The training script handles the model initialization, dataset preparation, training, validation, and evaluation pipeline.

---

## Why This Architecture?

The project combines several techniques to improve image restoration:

**MobileNetV3-Small**

Provides an efficient feature extraction backbone while keeping the encoder relatively lightweight.

**Multi-Scale Features**

Haze can affect images at different spatial scales. Multi-scale feature extraction allows the network to capture both detailed and larger contextual information.

**Latent Multi-Head Attention**

Attention allows the network to model relationships between extracted features and focus on useful information during reconstruction.

**Skip Connections**

Skip connections help preserve spatial information that may otherwise be lost during deeper feature processing.

**Residual Learning**

Instead of relying solely on direct image reconstruction, residual learning helps the network learn the transformation required to recover the clear image.

---

## Project Context

This project was developed as part of **Smart India Hackathon 2023**, focusing on the application of deep learning techniques to image restoration and visibility enhancement.

---

## Future Improvements

Potential improvements to the system include:

* More extensive benchmarking against existing dehazing architectures
* Additional real-world hazy image evaluation
* Model optimization for real-time inference
* Further experimentation with attention mechanisms
* Deployment as an image dehazing API or web application

---

## Author

**Aashay Metekar**

GitHub: [saturnextreme](https://github.com/saturnextreme)

---

## License

This project is available for educational and research purposes.
