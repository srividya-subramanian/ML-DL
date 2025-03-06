# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 05:59:18 2025

@author: srividya

🔹 Neural Network for Recognizing Direct Correlations
A Neural Network (NN) can learn direct correlations in data by mapping input features to output values using weights and activation functions. This example demonstrates how an NN can identify direct linear relationships and simple patterns in data.

🔹 Problem Statement
We create a neural network to recognize a simple direct correlation between input values and output labels.
For example:

Input: X = [2, 4, 6, 8]
Output: y = [4, 8, 12, 16] (Each y = 2 * X)
The network should learn the direct mapping function y = 2X without using explicit formulas.

🔹 Explanation
Weight (W) and Bias (b) Initialization:
    	Randomly initialize W and b before training.
Forward Pass:
    Compute predictions: y_pred = X * W + b
Loss Calculation (Mean Squared Error - MSE):
    Measure how far predictions are from true values.
Backpropagation:
    Compute gradients of W and b to minimize the error.
Gradient Descent Update:
    Adjust W and b based on learning rate.

🔹 Expected Output
The neural network will learn that the correlation between X and y is a factor of 2 (i.e., y ≈ 2 * X). After training, the final weight (W) should be close to 2, and the bias (b) should be close to 0.

"""

import numpy as np

# Generate synthetic data
X = np.array([[2], [4], [6], [8], [10]])  # Inputs
y = np.array([[4], [8], [12], [16], [20]])  # Outputs (Direct correlation: y = 2*X)

# Initialize weights and bias
W = np.random.randn(1, 1)  # Single weight
b = np.zeros((1, 1))  # Bias
learning_rate = 0.01
epochs = 1000

# Activation function (Linear Regression, no activation needed)
def forward(X):
    return np.dot(X, W) + b  # Linear transformation

# Mean Squared Error (MSE) loss
def loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

# Training loop (Gradient Descent)
for epoch in range(epochs):
    # Forward pass
    y_pred = forward(X)
    
    # Compute loss
    loss_value = loss(y_pred, y)
    
    # Compute gradients (derivative of MSE)
    dW = np.dot(X.T, (y_pred - y)) / len(X)
    db = np.mean(y_pred - y)
    
    # Update weights and bias
    W -= learning_rate * dW
    b -= learning_rate * db

    # Print progress every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss_value:.5f}")

# Final learned weight and bias
print("\nFinal Learned Parameters:")
print(f"Weight: {W.flatten()[0]:.5f}, Bias: {b.flatten()[0]:.5f}")

# Test the model
X_test = np.array([[12], [14]])
y_test_pred = forward(X_test)
print("\nPredictions for new inputs:")
print(y_test_pred)


'''
🔹 Summary
✅ Recognized Direct Correlation (y = 2X)
✅ Implemented Neural Network without Libraries (NumPy only)
✅ Used Gradient Descent to Learn the Pattern
✅ Achieved Accurate Predictions

'''

