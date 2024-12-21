Here’s a comprehensive README that explains your program and includes the mathematical formulas used:

---

# RandriaMlp: A Multi-Layer Perceptron Implementation

This repository contains a Python implementation of a Multi-Layer Perceptron (MLP) class, `RandriaMlp`, designed for binary classification tasks. The implementation includes functions for initialization, forward propagation, backpropagation, parameter updates, cost calculation, and predictions.

## Features
- **Cost Function**: Binary Cross-Entropy Loss
- **Forward Propagation**: Activation via Sigmoid Function
- **Backpropagation**: Gradient calculation for weights and biases
- **Training**: Gradient Descent Optimization
- **Visualization**: Training loss curve

---

## Mathematical Background

### 1. Cost Function: Binary Cross-Entropy Loss

The cost function measures the difference between predicted and actual values. For binary classification, the Binary Cross-Entropy Loss is given by:

\[
L = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(A^{(i)} + \epsilon) + (1 - y^{(i)}) \log(1 - A^{(i)} + \epsilon) \right]
\]

Where:
- \( y^{(i)} \) is the true label (0 or 1).
- \( A^{(i)} \) is the predicted probability.
- \( m \) is the number of samples.
- \( \epsilon \) is a small value to prevent numerical instability (\( \epsilon = 10^{-4} \)).

---

### 2. Forward Propagation

Forward propagation computes activations layer by layer using the sigmoid activation function:

\[
Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}
\]
\[
A^{[l]} = \frac{1}{1 + e^{-Z^{[l]}}}
\]

Where:
- \( Z^{[l]} \): Linear combination of inputs at layer \( l \).
- \( W^{[l]} \): Weight matrix for layer \( l \).
- \( b^{[l]} \): Bias vector for layer \( l \).
- \( A^{[l]} \): Activation at layer \( l \).

---

### 3. Backpropagation

Gradients of the cost function with respect to weights and biases are computed using the chain rule:

\[
\delta Z^{[l]} = A^{[l]} - y \quad \text{(for the last layer only)}
\]
\[
\delta W^{[l]} = \frac{1}{m} \delta Z^{[l]} (A^{[l-1]})^T
\]
\[
\delta b^{[l]} = \frac{1}{m} \sum \delta Z^{[l]}
\]
\[
\delta Z^{[l-1]} = W^{[l]^T} \delta Z^{[l]} \cdot A^{[l-1]} \cdot (1 - A^{[l-1]})
\]

Where:
- \( \delta Z^{[l]} \): Gradient of the loss with respect to \( Z^{[l]} \).
- \( \delta W^{[l]} \): Gradient of the loss with respect to \( W^{[l]} \).
- \( \delta b^{[l]} \): Gradient of the loss with respect to \( b^{[l]} \).

---

### 4. Weight and Bias Updates

Using Gradient Descent:

\[
W^{[l]} = W^{[l]} - \eta \delta W^{[l]}
\]
\[
b^{[l]} = b^{[l]} - \eta \delta b^{[l]}
\]

Where \( \eta \) is the learning rate.

---

### 5. Prediction

Predicted labels are obtained as:

\[
\hat{y} = 
\begin{cases} 
1 & \text{if } A^{[L]} \geq 0.5 \\
0 & \text{otherwise}
\end{cases}
\]

Where \( A^{[L]} \) is the activation of the last layer.

---

## Program Flow

1. **Initialization**: 
   - Random initialization of weights and biases for each layer.

2. **Training**:
   - Perform forward propagation to compute activations.
   - Compute the cost using Binary Cross-Entropy Loss.
   - Perform backpropagation to compute gradients.
   - Update parameters using Gradient Descent.

3. **Prediction**:
   - Use forward propagation to predict labels based on input data.

4. **Visualization**:
   - Plot the training loss over iterations.

---

## Usage

### Inputs
- `X`: Input feature matrix (\( n_{features} \times m \)).
- `y`: Ground truth labels (\( 1 \times m \)).
- `listrnn`: List of hidden layer dimensions (e.g., `[5, 3]` for a 2-layer model).
- `lr`: Learning rate (default = 0.1).
- `n`: Number of training iterations (default = 1000).

### Example
```python
X = np.array([[...]])  # Input features
y = np.array([[...]])  # True labels
listrnn = [4, 3]       # Hidden layer dimensions

parameters = RandriaMlp.artificial_neuron(X, y, listrnn, lr=0.01, n=1000)
```

### Output
- Final model parameters.
- Training loss curve.
- Predictions after training.

---

## Requirements
- Python 3.x
- NumPy
- Matplotlib

---

## Author
This implementation is designed by **Liantsoa RANDRIANASIMBOLARIVELO**.

---

This README provides a clear understanding of the program and its underlying mathematical principles. Let me know if you need further refinements!# RandriaMlp: A Multi-Layer Perceptron Implementation

This repository contains a Python implementation of a Multi-Layer Perceptron (MLP) class, `RandriaMlp`, designed for binary classification tasks. The implementation includes functions for initialization, forward propagation, backpropagation, parameter updates, cost calculation, and predictions.

## Features
- **Cost Function**: Binary Cross-Entropy Loss
- **Forward Propagation**: Activation via Sigmoid Function
- **Backpropagation**: Gradient calculation for weights and biases
- **Training**: Gradient Descent Optimization
- **Visualization**: Training loss curve

---

## Mathematical Background

### 1. Cost Function: Binary Cross-Entropy Loss

The cost function measures the difference between predicted and actual values. For binary classification, the Binary Cross-Entropy Loss is given by:

\[
L = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(A^{(i)} + \epsilon) + (1 - y^{(i)}) \log(1 - A^{(i)} + \epsilon) \right]
\]

Where:
- \( y^{(i)} \) is the true label (0 or 1).
- \( A^{(i)} \) is the predicted probability.
- \( m \) is the number of samples.
- \( \epsilon \) is a small value to prevent numerical instability (\( \epsilon = 10^{-4} \)).

---

### 2. Forward Propagation

Forward propagation computes activations layer by layer using the sigmoid activation function:

\[
Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}
\]
\[
A^{[l]} = \frac{1}{1 + e^{-Z^{[l]}}}
\]

Where:
- \( Z^{[l]} \): Linear combination of inputs at layer \( l \).
- \( W^{[l]} \): Weight matrix for layer \( l \).
- \( b^{[l]} \): Bias vector for layer \( l \).
- \( A^{[l]} \): Activation at layer \( l \).

---

### 3. Backpropagation

Gradients of the cost function with respect to weights and biases are computed using the chain rule:

\[
\delta Z^{[l]} = A^{[l]} - y \quad \text{(for the last layer only)}
\]
\[
\delta W^{[l]} = \frac{1}{m} \delta Z^{[l]} (A^{[l-1]})^T
\]
\[
\delta b^{[l]} = \frac{1}{m} \sum \delta Z^{[l]}
\]
\[
\delta Z^{[l-1]} = W^{[l]^T} \delta Z^{[l]} \cdot A^{[l-1]} \cdot (1 - A^{[l-1]})
\]

Where:
- \( \delta Z^{[l]} \): Gradient of the loss with respect to \( Z^{[l]} \).
- \( \delta W^{[l]} \): Gradient of the loss with respect to \( W^{[l]} \).
- \( \delta b^{[l]} \): Gradient of the loss with respect to \( b^{[l]} \).

---

### 4. Weight and Bias Updates

Using Gradient Descent:

\[
W^{[l]} = W^{[l]} - \eta \delta W^{[l]}
\]
\[
b^{[l]} = b^{[l]} - \eta \delta b^{[l]}
\]

Where \( \eta \) is the learning rate.

---

### 5. Prediction

Predicted labels are obtained as:

\[
\hat{y} = 
\begin{cases} 
1 & \text{if } A^{[L]} \geq 0.5 \\
0 & \text{otherwise}
\end{cases}
\]

Where \( A^{[L]} \) is the activation of the last layer.

---

## Program Flow

1. **Initialization**: 
   - Random initialization of weights and biases for each layer.

2. **Training**:
   - Perform forward propagation to compute activations.
   - Compute the cost using Binary Cross-Entropy Loss.
   - Perform backpropagation to compute gradients.
   - Update parameters using Gradient Descent.

3. **Prediction**:
   - Use forward propagation to predict labels based on input data.

4. **Visualization**:
   - Plot the training loss over iterations.

---

## Usage

### Inputs
- `X`: Input feature matrix (\( n_{features} \times m \)).
- `y`: Ground truth labels (\( 1 \times m \)).
- `listrnn`: List of hidden layer dimensions (e.g., `[5, 3]` for a 2-layer model).
- `lr`: Learning rate (default = 0.1).
- `n`: Number of training iterations (default = 1000).

### Example
```python
X = np.array([[...]])  # Input features
y = np.array([[...]])  # True labels
listrnn = [4, 3]       # Hidden layer dimensions

parameters = RandriaMlp.artificial_neuron(X, y, listrnn, lr=0.01, n=1000)
```

### Output
- Final model parameters.
- Training loss curve.
- Predictions after training.

---

## Requirements
- Python 3.x
- NumPy
- Matplotlib

---

## Author
This implementation is designed by **Liantsoa RANDRIANASIMBOLARIVELO**.

---
