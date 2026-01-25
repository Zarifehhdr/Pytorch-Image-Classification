# MNIST Handwritten Digit Classification using PyTorch

In this project, we use PyTorch and linear models to solve an image classification problem with the famous **MNIST Handwritten Digits Database**. This dataset contains 28×28 grayscale images of handwritten digits (0–9), along with labels indicating the digit each image represents.

---

## Dataset

The **MNIST dataset** consists of:

- **Images:** 28×28 grayscale images of handwritten digits
- **Labels:** Digit labels (0–9) corresponding to each image

Example images from the dataset:

![MNIST Sample](https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png)

> Each image is a single handwritten digit. Our goal is to train a model to correctly classify these digits.

---

## Project Steps

1. **Load and Explore the Dataset**  
   - Download MNIST using PyTorch's `torchvision.datasets`  
   - Visualize sample images and inspect their labels

2. **Prepare the Dataset**  
   - Convert images to tensors  
   - Normalize pixel values  
   - Split dataset into training and validation sets  
   - Use `DataLoader` for batching

3. **Build a Linear Model**  
   - Implement a simple linear classifier using `nn.Linear`  
   - Flatten 28×28 images into 784-dimensional input vectors

4. **Train the Model**  
   - Use a loss function (CrossEntropyLoss)  
   - Optimize model parameters with an optimizer like SGD or Adam  
   - Track training and validation accuracy

5. **Evaluate the Model**  
   - Calculate accuracy on the validation/test set  
   - Visualize predictions for sample images

---

## Requirements

- Python 3.8+  
- PyTorch  
- torchvision  
- matplotlib / seaborn (optional for visualization)  
- numpy  

Install dependencies with:

```bash
pip install torch torchvision matplotlib numpy
