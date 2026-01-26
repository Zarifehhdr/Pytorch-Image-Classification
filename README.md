# MNIST Handwritten Digit Classification using PyTorch

This repository demonstrates handwritten digit classification on the **MNIST dataset** using **PyTorch**.  
The project is implemented in **two Jupyter notebooks** to show how adding a hidden layer and nonlinearity improves model performance.

---

## 📌 Overview

- Dataset: **MNIST Handwritten Digits (0–9)**
- Image size: **28×28 grayscale**
- Framework: **PyTorch**
- Goal: Compare a **simple linear model** with a **nonlinear neural network**

---

## 🧠 Models Implemented

### 1️⃣ Simple Neural Network (Linear Model)

- Single `nn.Linear` layer  
- Input: 784 (flattened 28×28 image)  
- Output: 10 classes  
- No hidden layers or nonlinear activation  
- Equivalent to multinomial logistic regression  

📓 Notebook: `mnist_linear_model.ipynb`

---

### 2️⃣ Neural Network with Hidden Layer

- One hidden layer  
- Nonlinear activation (e.g., ReLU)  
- Learns more complex feature representations  
- Achieves **~15% higher accuracy** than the linear model  

📓 Notebook: `mnist_hidden_layer_model.ipynb`

---

## 📊 Dataset

The MNIST dataset contains:

- **60,000** training images  
- **10,000** test images  
- Each image is a **28×28 grayscale handwritten digit**

Example images:

![MNIST Sample](https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png)

---

## 🔧 Project Workflow

1. Load MNIST using `torchvision.datasets`
2. Preprocess images:
   - Convert to tensors
   - Normalize pixel values
3. Create `DataLoader` for batching
4. Define model architectures
5. Train using:
   - `CrossEntropyLoss`
   - SGD or Adam optimizer
6. Evaluate and compare model accuracy

---

## 📈 Results

| Model | Description | Accuracy |
|------|------------|----------|
| Linear Model | Single linear layer | Baseline |
| Hidden Layer Model | Nonlinear NN | **~15% higher** |

> Adding a hidden layer and nonlinearity significantly improves classification accuracy, even with a small architectural change.

---

## 📦 Requirements

- Python 3.8+
- PyTorch
- torchvision
- numpy
- matplotlib (optional)

Install dependencies:

```bash
pip install torch torchvision numpy matplotlib
