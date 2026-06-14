# E-Commerce Clothing Classifier with CNN

Building a **Convolutional Neural Network (CNN)** to automatically categorize clothing images for **Fashion Forward** — an AI-driven e-commerce retailer — enabling automated product listing and smarter inventory management.

---

## Problem Statement

Manually categorizing product listings is slow and error-prone at scale. This project trains a CNN on Fashion MNIST to classify garment images into 10 categories, powering automatic product tagging for e-commerce platforms.

---

## Dataset

**File:** `data/fashion_mnist.npz` — Fashion MNIST dataset

| Split | Samples | Shape |
|---|---|---|
| Train | 60,000 | 28 × 28 grayscale |
| Test | 10,000 | 28 × 28 grayscale |

**10 clothing categories:**

| Label | Class |
|---|---|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

---

## Approach

### 1. Preprocessing
- Pixel normalization: `[0, 255]` → `[0.0, 1.0]`
- Reshape: `(N, 28, 28)` → `(N, 28, 28, 1)` to add channel dimension
- One-hot encoding of labels (10 classes)

### 2. Model Architecture — CNN (Sequential)

```
Input: (28, 28, 1)
  └─► Conv2D(32, kernel=3, activation='relu')
  └─► Conv2D(16, kernel=3, activation='relu')
  └─► Flatten
  └─► Dense(10, activation='softmax')
```

### 3. Training

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Loss | Categorical Crossentropy |
| Epochs | 1 |
| Batch size | 64 |

### 4. Results

| Metric | Value |
|---|---|
| Train Accuracy | 81.34% |
| **Test Accuracy** | **85.36%** |

> Achieved **85.36% test accuracy in a single epoch**, demonstrating the efficiency of CNNs for image classification even with minimal training.

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.8-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Sequential_API-D00000?logo=keras&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Data-013243?logo=numpy)

- **TensorFlow / Keras** — model definition, training, and evaluation
- **NumPy** — data loading and preprocessing
- **Matplotlib** — dataset visualization

---

## Project Structure

```
E-Commerce Clothing Classifier/
│
├── data/
│   └── fashion_mnist.npz
├── images/
│   └── Clothing Classifier Model.png
├── notebook.ipynb
└── README.md
```

---

## Key Takeaways

- A lightweight 2-layer CNN achieves strong baseline accuracy (85%+) on Fashion MNIST in just one epoch, making it viable for rapid prototyping in e-commerce product categorization.
- Automated image classification reduces manual labeling overhead and enables consistent product tagging at scale.
- Further gains are possible with additional epochs, data augmentation, batch normalization, or deeper architectures (e.g. ResNet, MobileNet).
