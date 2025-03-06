# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:59:14 2025

@author: srivi
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml  # Used only for fetching MNIST data
from sklearn.model_selection import train_test_split

# 1️⃣ Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist.data.astype(np.float32), mnist.target.astype(np.int32)

# 2️⃣ Normalize data (scale pixel values to 0-1)
X /= 255.0  

# 3️⃣ Convert labels to one-hot encoding
y_onehot = np.eye(10)[y]

# 4️⃣ Split dataset into training (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

# 5️⃣ Define Neural Network Parameters
input_size = 784  # 28x28 pixels
hidden_size = 128
output_size = 10  # 10 classes (digits 0-9)
learning_rate = 0.01
l2_lambda = 0.001  # L2 Regularization strength
epochs = 30
batch_size = 64  # Selectable batch size

# Initialize weights and biases
np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size) * 0.01  # Small random values
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))

# Activation functions
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Stability trick
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Derivative of ReLU
def relu_derivative(x):
    return (x > 0).astype(float)

# Cross-entropy loss with L2 regularization
def cross_entropy_loss(y_pred, y_true):
    loss = -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))
    l2_penalty = (l2_lambda / 2) * (np.sum(W1**2) + np.sum(W2**2))  # L2 Regularization term
    return loss + l2_penalty

# Mini-batch gradient descent training
num_samples = X_train.shape[0]
num_batches = num_samples // batch_size

for epoch in range(epochs):
    shuffled_indices = np.random.permutation(num_samples)
    X_train_shuffled = X_train[shuffled_indices]
    y_train_shuffled = y_train[shuffled_indices]

    for batch in range(num_batches):
        # Extract mini-batch
        start = batch * batch_size
        end = start + batch_size
        X_batch = X_train_shuffled[start:end]
        y_batch = y_train_shuffled[start:end]

        # 🔹 Forward Pass
        Z1 = X_batch @ W1 + b1
        A1 = relu(Z1)
        Z2 = A1 @ W2 + b2
        A2 = softmax(Z2)

        # 🔹 Compute Loss with L2 Regularization
        loss = cross_entropy_loss(A2, y_batch)

        # 🔹 Backpropagation
        dZ2 = A2 - y_batch
        dW2 = (A1.T @ dZ2) / batch_size + l2_lambda * W2
        db2 = np.mean(dZ2, axis=0, keepdims=True)

        dA1 = dZ2 @ W2.T
        dZ1 = dA1 * relu_derivative(Z1)
        dW1 = (X_batch.T @ dZ1) / batch_size + l2_lambda * W1
        db1 = np.mean(dZ1, axis=0, keepdims=True)

        # 🔹 Gradient Descent Update
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

    # Print progress every 5 epochs
    if epoch % 5 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# 6️⃣ Evaluate Model on Test Set
Z1_test = X_test @ W1 + b1
A1_test = relu(Z1_test)
Z2_test = A1_test @ W2 + b2
A2_test = softmax(Z2_test)

# Convert probabilities to class labels
y_pred = np.argmax(A2_test, axis=1)
y_true = np.argmax(y_test, axis=1)

# Compute accuracy
accuracy = np.mean(y_pred == y_true)
print(f"Test Accuracy: {accuracy:.4f}")

# 7️⃣ Display some predictions
fig, axes = plt.subplots(1, 5, figsize=(10, 5))
for i, ax in enumerate(axes):
    ax.imshow(X_test[i].reshape(28, 28), cmap='gray')
    ax.set_title(f"Pred: {y_pred[i]}")
    ax.axis('off')
plt.show()