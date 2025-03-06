# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 08:15:38 2025

@author: srivi
"""

import numpy as np
import cv2
import os
import random

# Load dataset (assumes images are stored in 'cats/' and 'dogs/' folders)
def load_images(folder, label, img_size=(64, 64)):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
        if img is not None:
            img = cv2.resize(img, img_size)  # Resize to (64, 64)
            images.append(img.flatten() / 255.0)  # Normalize and flatten
            labels.append(label)
    return images, labels

# Load cat and dog images
cat_images, cat_labels = load_images("cats/", label=0)
dog_images, dog_labels = load_images("dogs/", label=1)

# Combine datasets and shuffle
X = np.array(cat_images + dog_images)
y = np.array(cat_labels + dog_labels).reshape(-1, 1)

# Shuffle data
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X, y = X[indices], y[indices]

# Train-test split (80% train, 20% test)
split = int(0.8 * X.shape[0])
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Define activation functions
def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(Z):
    return (Z > 0).astype(float)  # 1 for positive values, 0 otherwise

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def sigmoid_derivative(Z):
    return Z * (1 - Z)

# Initialize network parameters
input_size = X_train.shape[1]  # 64x64 pixels flattened
hidden_size = 128  # Hidden layer neurons
output_size = 1  # Binary classification (cat/dog)
learning_rate = 0.01
epochs = 1000

# Random weight initialization
np.random.seed(42)
W1 = np.random.randn(hidden_size, input_size) * 0.01
b1 = np.zeros((hidden_size, 1))
W2 = np.random.randn(output_size, hidden_size) * 0.01
b2 = np.zeros((output_size, 1))

# Training the neural network
for epoch in range(epochs):
    # Forward propagation
    Z1 = np.dot(W1, X_train.T) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    # Compute loss (binary cross-entropy)
    m = y_train.shape[0]
    loss = (-1 / m) * np.sum(y_train.T * np.log(A2) + (1 - y_train.T) * np.log(1 - A2))

    # Backpropagation
    dZ2 = A2 - y_train.T
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = (1 / m) * np.dot(dZ1, X_train)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    # Update weights
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    # Print loss every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Prediction function
def predict(X):
    Z1 = np.dot(W1, X.T) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    return (A2 > 0.5).astype(int).flatten()

# Evaluate accuracy
y_pred = predict(X_test)
accuracy = np.mean(y_pred == y_test.flatten()) * 100
print(f"Test Accuracy: {accuracy:.2f}%")