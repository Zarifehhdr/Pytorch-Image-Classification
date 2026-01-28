# Image Classification using PyTorch

This repository demonstrates handwritten digit classification on **two MNIST-based datasets** using **PyTorch**.  
The project is implemented in **multiple Jupyter notebooks** to show how adding a hidden layer and nonlinearity improves model performance, and how the same workflow generalizes to a more challenging dataset.

---

## 📌 Overview

- Datasets:
  - **MNIST Handwritten Digits (0–9)**
  - **Fashion-MNIST (10 clothing categories)**
- Image size: **28×28 grayscale**
- Framework: **PyTorch**
- Goal: Compare a **simple linear model** with a **nonlinear neural network**, and demonstrate performance differences across datasets

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
- Achieves **~15% higher accuracy** than the linear model on MNIST  

📓 Notebook: `mnist_hidden_layer_model.ipynb`

---

### 3️⃣ Image Classification on Fashion-MNIST

- Uses the **Fashion-MNIST** dataset (Zalando clothing images)
- Same image size and format as MNIST (28×28 grayscale)
- 10 classes (e.g., T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)
- Demonstrates how the same model architectures:
  - Transfer directly to a new dataset
  - Face a more challenging classification task due to higher visual similarity between classes
- Highlights the importance of **nonlinearity and representation learning** even more strongly than digit classification

📓 Notebook: `fashion_mnist_classification.ipynb`

---

## 📊 Datasets

### MNIST
The MNIST dataset contains:

- **60,000** training images  
- **10,000** test images  
- Each image is a **28×28 grayscale handwritten digit**

Example images:

![MNIST Sample](https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png)

---

### Fashion-MNIST

The Fashion-MNIST dataset contains:

- **60,000** training images  
- **10,000** test images  
- Each image is a **28×28 grayscale clothing item**
- Designed as a more difficult, drop-in replacement for MNIST

---

## 🔧 Project Workflow

1. Load datasets using `torchvision.datasets`
2. Preprocess images:
   - Convert to tensors
   - Normalize pixel values
3. Create `DataLoader` for batching
4. Define model architectures
5. Train using:
   - `CrossEntropyLoss`
   - SGD or Adam optimizer
6. Evaluate and compare model accuracy across datasets and architectures

---

## 📈 Results

| Model | Dataset | Description | Accuracy |
|------|--------|-------------|----------|
| Linear Model | MNIST | Single linear layer | Baseline |
| Hidden Layer Model | MNIST | Nonlinear NN | **~15% higher** |
| Linear / Hidden Layer | Fashion-MNIST | Same architectures | Lower overall accuracy, clearer benefit from nonlinearity |

> Adding a hidden layer and nonlinearity significantly improves classification accuracy, and the effect is even more pronounced on Fashion-MNIST due to its higher visual complexity.

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
