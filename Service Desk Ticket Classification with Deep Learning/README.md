# Service Desk Ticket Classification with Deep Learning

> **NLP · CNN · Text Classification · PyTorch**

A deep learning pipeline that automatically categorizes customer complaints into 5 financial service categories — built for **CleverSupport**, a fictional AI-driven customer support company. No pre-trained models, no shortcuts: a CNN text classifier trained from scratch.

---

## 🧩 Problem Statement

Customer support teams at financial institutions are overwhelmed by high volumes of complaints. Manual categorization is slow and inconsistent. This project builds an end-to-end text classifier that automatically routes complaints to the right department — reducing response time and improving support quality.

**Categories:** Mortgage · Credit Card · Money Transfers · Debt Collection · Other Financial Services

---

## 📊 Dataset

| Property | Value |
|---|---|
| Total samples | 5,000 complaints |
| Classes | 5 (perfectly balanced — 1,000 per class) |
| Vocabulary size | 10,146 unique words |
| Sequence length | Padded to 50 tokens |
| Train / Test split | 80% / 20% |

---

## 🏗️ Model Architecture

A custom **1D Convolutional Neural Network (CNN)** built with PyTorch — no pre-trained embeddings, trained end-to-end from scratch.

```
Input (seq_len=50)
    → Embedding Layer (vocab=10,146 → dim=64)
    → Conv1D (128 filters, kernel=3) + ReLU
    → Global Max Pooling
    → Fully Connected (128 → 5 classes)
    → Output (CrossEntropyLoss)
```

**Hyperparameters:**

| Parameter | Value |
|---|---|
| Embedding dim | 64 |
| CNN filters | 128 |
| Kernel size | 3 |
| Batch size | 32 |
| Epochs | 3 |
| Optimizer | Adam (lr=1e-3) |

---

## 📈 Results

| Epoch | Loss |
|---|---|
| 1 | 1.3174 |
| 2 | 0.7629 |
| 3 | 0.5741 |

**Test Set Performance (3 epochs):**

| Metric | Score |
|---|---|
| **Accuracy** | **75.60%** |

| Class | Precision | Recall |
|---|---|---|
| Class 0 (Mortgage) | 71.1% | 69.3% |
| Class 1 (Credit Card) | 77.2% | 69.5% |
| Class 2 (Money Transfers) | 81.2% | 74.1% |
| Class 3 (Debt Collection) | 63.2% | 82.3% |
| Class 4 (Other) | 88.7% | 82.4% |

> Training loss dropped **56% from epoch 1 to 3**, showing clear convergence. Accuracy of 75.6% achieved in just 3 epochs on a balanced 5-class problem — further improvement expected with more epochs or pre-trained embeddings (e.g. GloVe).

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-3776AB?style=flat&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)

---

## 📁 Project Structure

```
Service Desk Ticket Classification with Deep Learning/
├── notebook.ipynb    # Full pipeline: preprocessing → model → evaluation
├── text.json         # Tokenized complaint texts
├── words.json        # Vocabulary (10,146 words)
├── labels.npy        # Encoded class labels (0–4)
└── servicedesk.png   # Project cover
```

---

## 💡 What I Learned

- Building a **CNN text classifier from scratch** in PyTorch without pre-trained models
- Implementing **sequence padding** and custom word-to-index mappings for NLP preprocessing
- Using **global max pooling** after Conv1D to capture the most prominent features regardless of position
- Evaluating multiclass models with **per-class precision and recall** using `torchmetrics`
- Understanding the trade-off between recall and precision across imbalanced predictions (Class 3 shows high recall but lower precision)
