# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 09:37:29 2025

@author: srivi
"""

import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self._backward = lambda: None  # Placeholder for backward function

    def backward(self):
        if self.requires_grad:
            self._backward()

class LSTM:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size

        # Learnable parameters (with autograd)
        self.Wf = Tensor(np.random.randn(hidden_size, input_size + hidden_size) * 0.01, requires_grad=True)
        self.Wi = Tensor(np.random.randn(hidden_size, input_size + hidden_size) * 0.01, requires_grad=True)
        self.Wo = Tensor(np.random.randn(hidden_size, input_size + hidden_size) * 0.01, requires_grad=True)
        self.Wc = Tensor(np.random.randn(hidden_size, input_size + hidden_size) * 0.01, requires_grad=True)
        self.Why = Tensor(np.random.randn(output_size, hidden_size) * 0.01, requires_grad=True)

        # Initial states
        self.h = np.zeros((hidden_size, 1))
        self.c = np.zeros((hidden_size, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        combined = np.vstack((self.h, x))  # Combine previous hidden state and input
        
        forget_gate = self.sigmoid(np.dot(self.Wf.data, combined))
        input_gate = self.sigmoid(np.dot(self.Wi.data, combined))
        output_gate = self.sigmoid(np.dot(self.Wo.data, combined))
        cell_candidate = np.tanh(np.dot(self.Wc.data, combined))
        
        self.c = forget_gate * self.c + input_gate * cell_candidate  # Update cell state
        self.h = output_gate * np.tanh(self.c)  # Update hidden state
        
        y = np.dot(self.Why.data, self.h)  # Output
        return Tensor(y, requires_grad=True), Tensor(self.h, requires_grad=True)


def backward(tensor, grad):
    if tensor.requires_grad:
        tensor.grad += grad  # Accumulate gradients
        tensor._backward()  # Call the backward function

def step(parameters, lr=0.01):
    for param in parameters:
        if param.requires_grad:
            param.data -= lr * param.grad  # Update weights
            param.grad.fill(0)  # Reset gradients after update

# Training data (one-hot encoded words)
X_train = [
    np.array([[0], [1], [0], [0], [0]]),  # "blue"
    np.array([[1], [0], [1], [0], [0]])   # "cloudy"
]
Y_train = [
    np.array([[1], [0], [0], [0], [0]]),  # Target: "cloudy"
    np.array([[0], [1], [0], [0], [0]])   # Target: "sunny"
]

# Initialize LSTM
lstm = LSTM(input_size=5, hidden_size=10, output_size=5)

epochs = 100
learning_rate = 0.01

for epoch in range(epochs):
    total_loss = 0

    for x, y_true in zip(X_train, Y_train):
        # Forward pass
        y_pred, h = lstm.forward(Tensor(x, requires_grad=True))

        # Compute loss (Mean Squared Error)
        loss = np.mean((y_pred.data - y_true) ** 2)
        total_loss += loss

        # Backpropagate gradients
        grad_y = 2 * (y_pred.data - y_true) / y_true.size
        backward(y_pred, grad_y)
        
        # Update weights
        step([lstm.Wf, lstm.Wi, lstm.Wo, lstm.Wc, lstm.Why], lr=learning_rate)

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

