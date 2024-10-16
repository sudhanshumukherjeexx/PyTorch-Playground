# PyTorch-Playground

Welcome to **PyTorch Playground**, a repository dedicated to showcasing a wide array of PyTorch concepts and operations. This repository contains hands-on tutorials, code examples, and best practices aimed at helping you master PyTorch from the ground up.

## Repository Overview

The goal of this repository is to provide a practical, beginner-to-advanced learning experience in PyTorch. Through a series of notebooks, we will explore different facets of deep learning, tensor operations, and neural networks, all implemented in PyTorch.

### Featured Notebooks:
- **[🔗 Notebook 1: Exploring PyTorch Tensor Operations](https://colab.research.google.com/drive/1zxiKzcRWMQ2ukA50v6V9cPjrsHfSiuUn?usp=sharing)**
- **[🔗 Notebook 2: Building Neural Network from Scratch](https://colab.research.google.com/drive/1UeV8DGYC6vUjRXijdMWQt2z0SdgRx3v0?usp=sharing)**  
- [Future Notebooks] (coming soon)

---
🧠 `Click to expland for details`
<details>
  <summary>Notebook 1: Exploring PyTorch Tensor Operations</summary>

This notebook introduces the foundational concepts of tensor operations in PyTorch. Tensors are the basic building blocks for deep learning models, and this notebook covers how to initialize, manipulate, and operate on tensors effectively.

### Key Topics Covered:

#### Initializing Tensors:
Learn how to create 1D, 2D, and 3D tensors, as well as how to initialize tensors with specific values or random values.

- 1D, 2D, and 3D tensor creation
- Randomly initialized tensors
- Tensors with specific data types and values

#### Mathematical Operations:
Perform element-wise operations, broadcasting, and basic mathematical functions on tensors.

- Element-wise operations (addition, subtraction, multiplication, and division)
- Exponentiation and logarithms
- Summation, mean, min, and max operations
- Broadcasting in tensor operations

#### Matrix Operations:
Explore essential matrix operations like addition, subtraction, multiplication (dot product), transposing, reshaping, and slicing.

- Matrix addition, subtraction, and multiplication
- Matrix transpose and inverse
- Reshaping and slicing tensors
- Matrix determinant

#### Tensor Concatenation and Stacking:
Combine tensors using concatenation and stacking operations for flexible data manipulation.

#### Conversion between PyTorch Tensors and NumPy Arrays:
Seamlessly convert between PyTorch tensors and NumPy arrays for compatibility with the broader Python ecosystem.

#### Automatic Differentiation:
A brief introduction to PyTorch's automatic differentiation functionality using `requires_grad` and `backward()`.

</details>
<details>
  <summary>Notebook 2: Building Neural Network from Scratch</summary>

This notebook walks you through the process of building a neural network from scratch using PyTorch. It covers essential steps like loading a dataset, designing the network architecture, and implementing forward and backward propagation.

### Key Topics Covered:

#### Loading the Dataset:
We use the MNIST dataset for real-life image classification. You will learn:
- How to load the dataset with PyTorch’s `DataLoader`
- How to preprocess and normalize the dataset for better model performance
- Visualizing sample data to understand the input-output structure

#### Architecture of the Neural Network:
Understand how to define and build a fully connected neural network with input, hidden, and output layers.
- Defining input, hidden, and output neurons
- Implementing the architecture with PyTorch’s `nn.Module`
- Applying activation functions like ReLU and Softmax

#### Initializing Weights:
We explore how to initialize weights for the network:
- PyTorch’s default weight initialization
- Manual initialization using `torch.nn.init` methods for more control

#### Forward Propagation:
Implementing forward propagation to compute the output given the input:
- Flattening image data for input
- Applying activation functions between layers
- Computing the output using logits

#### Backward Propagation:
Using PyTorch’s automatic differentiation to compute the gradients and update the weights:
- Calculating the loss with `CrossEntropyLoss`
- Applying backpropagation with `loss.backward()`
- Updating weights with gradient descent using an optimizer (SGD/Adam)

#### Training the Model for `n` Epochs:
Train the neural network and observe how the loss decreases over time:
- Iterating over multiple epochs and mini-batches
- Printing the loss at each epoch to monitor training
- Evaluating the model’s performance on the test set

#### Visualizing Error Loss:
After training, visualize the error loss per epoch to understand the model’s learning process:
- Plotting the loss curve using `matplotlib`
- Analyzing the network's convergence

#### Evaluating the Model:
Evaluate the trained model on test data and display the accuracy:
- Compute the accuracy of the model
- Visualizing predictions vs. actual results on sample images

</details>
---
#### 🚀 How to Run the Notebook:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/pytorch-playground.git
   ```

2. Open the first notebook:
   ```bash
   cd pytorch-playground
   ```

3. Run the notebook in Google Colab or any local environment with PyTorch installed.

---

#### Installation and Requirements

This repository uses PyTorch. If you’re running it locally, make sure you have PyTorch installed:
```bash
pip install torch torchvision matplotlib numpy
```

Alternatively, you can run the notebooks in [Google Colab](https://colab.research.google.com/), where no setup is required, as PyTorch comes pre-installed.

---
#### Future Plans

Stay tuned for more exciting notebooks on:
- Convolutional Neural Networks (CNNs)
- Transfer Learning with PyTorch
- RNNs and LSTMs for Sequence Modeling
- Autoencoders and Generative Models
