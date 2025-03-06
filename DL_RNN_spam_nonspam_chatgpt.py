# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 14:25:40 2025

@author: srivi
"""

'''
Neural Network
How It Works:
Processes data in one direction, from input to output.
No memory of previous inputs; each input is processed independently.

Architecture:
Input Layer → Hidden Layers (with activation functions) → Output Layer
Example: Multi-Layer Perceptron (MLP)

Use Cases:
✅ Image Classification (e.g., detecting objects in images)
✅ Tabular Data Analysis (e.g., predicting house prices)
✅ Sentiment Analysis (Basic Models)

Recurrent Neural Network

Unlike standard NNs, RNNs have memory:
They use previous hidden states as input for the next step.
Designed for sequential data (e.g., text, speech, time series).
Can capture dependencies between past and present inputs.

Architecture:
Each neuron loops back to process previous states.
Uses hidden states to retain past information.
Example: Simple RNN, LSTM, GRU

📌 Standard RNN Cell Update Formula
At each time step t, the hidden state is updated as:

ht = tanh(Wh ht−1 + Wx xt + b)
ht= Current hidden state
ht−1= Previous hidden state
xt= Current input
b = Bias⁡
tanh = Activation function
The final output is computed as:
yt=σ(Wy ht + by)
​



Use Cases:
✅ Text Processing (NLP) (e.g., machine translation, text generation)
✅ Speech Recognition (e.g., voice assistants)
✅ Stock Price Prediction (e.g., financial time series)


🔹 Key Differences
Feature		     Standard Neural Network (FNN)	Recurrent Neural Network (RNN)
Data Type	       Independent inputs	        Sequential inputs (time-dependent)
Memory	           No memory	                Retains past information
Backpropagation	   Standard	                    Uses Backpropagation Through Time (BPTT)
Best For	       Images, tabular data	        Text, speech, time series
Limitations	      Cannot handle sequences well	Struggles with long-term dependencies 
                                                (solved with LSTMs/GRUs)

🔹 Example: Comparing FNN vs. RNN for Text Classification
Dataset: Simple binary classification (spam vs. non-spam messages).
📌 Steps in both models:
✅ Tokenization + One-Hot Encoding
✅ Feedforward Neural Network (FNN)
✅ Recurrent Neural Network (RNN)
✅ Training using Batch Gradient Descent

'''

import numpy as np

# Sample dataset (simplified text messages)
messages = ["hello how are you", "win money now", "cheap loans available", "good morning friend"]
labels = np.array([0, 1, 1, 0])  # 0 = Non-Spam, 1 = Spam

# Simple word tokenization (creating vocabulary)
word_list = sorted(set(" ".join(messages).split()))
word_index = {word: i for i, word in enumerate(word_list)}

# Convert text to one-hot encoding
def encode_message(msg):
    vector = np.zeros(len(word_list))
    for word in msg.split():
        if word in word_index:
            vector[word_index[word]] = 1
    return vector

X = np.array([encode_message(msg) for msg in messages])
y = labels.reshape(-1, 1)  # Reshape for compatibility



# Initialize weights for FNN
input_size = len(word_list)
hidden_size = 4  # Small hidden layer
output_size = 1  # Binary classification

W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

# ReLU activation function
def relu(x):
    return np.maximum(0, x)

# Sigmoid for binary classification
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Forward Pass for FNN
def forward_fnn(X):
    hidden = relu(np.dot(X, W1) + b1)
    output = sigmoid(np.dot(hidden, W2) + b2)
    return output

# Predictions
y_pred_fnn = forward_fnn(X)
print("\n🔹 FNN Predictions:", np.round(y_pred_fnn).flatten())


# RNN Parameters
hidden_size_rnn = 4  # Hidden state size
W_xh = np.random.randn(input_size, hidden_size_rnn)  # Input to hidden weights
W_hh = np.random.randn(hidden_size_rnn, hidden_size_rnn)  # Hidden to hidden weights
W_hy = np.random.randn(hidden_size_rnn, output_size)  # Hidden to output weights
b_h = np.zeros((1, hidden_size_rnn))  # Hidden bias
b_y = np.zeros((1, output_size))  # Output bias
learning_rate = 0.01

# Tanh activation for RNN hidden state

# Activation functions
def tanh(x):
    return np.tanh(x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivatives for backpropagation
def dtanh(x):
    return 1 - np.tanh(x) ** 2

def dsigmoid(x):
    return x * (1 - x)

# RNN Forward Pass
# Forward pass for one sample
def forward_rnn(X_sample):
    h_t = np.zeros((1, hidden_size))  # Initial hidden state
    hidden_states = []

    # Process each word position (sequence)
    for t in range(X_sample.shape[1]):  
        h_t = tanh(np.dot(X_sample[:, t:t+1], W_xh[t:t+1, :]) + np.dot(h_t, W_hh) + b_h)
        hidden_states.append(h_t)
    
    hidden_states = np.array(hidden_states)
    output = sigmoid(np.dot(h_t, W_hy) + b_y)
    
    return output, hidden_states

# Backpropagation Through Time (BPTT)
def backward_rnn(X_sample, y_sample, output, hidden_states):
    global W_xh, W_hh, W_hy, b_h, b_y
    
    # Compute output error
    error = output - y_sample
    d_output = error * dsigmoid(output)  # Gradient of output layer
    
    # Gradients initialization
    dW_hy = np.dot(hidden_states[-1].T, d_output)
    db_y = np.sum(d_output, axis=0, keepdims=True)
    
    # Initialize gradients for hidden layers
    dW_xh = np.zeros_like(W_xh)
    dW_hh = np.zeros_like(W_hh)
    db_h = np.zeros_like(b_h)
    
    dh_next = np.dot(d_output, W_hy.T)
    
    # Backpropagate through time
    for t in reversed(range(len(hidden_states))):
        dh = dh_next * dtanh(hidden_states[t])  # Apply tanh derivative
        dW_xh += np.dot(X_sample[:, t:t+1].T, dh)
        dW_hh += np.dot(hidden_states[t-1].T if t > 0 else np.zeros_like(hidden_states[t].T), dh)
        db_h += dh
        dh_next = np.dot(dh, W_hh.T)  # Backpropagate to previous hidden state

    # Update weights
    W_xh -= learning_rate * dW_xh
    W_hh -= learning_rate * dW_hh
    W_hy -= learning_rate * dW_hy
    b_h -= learning_rate * db_h
    b_y -= learning_rate * db_y



# Training loop
epochs = 4000
for epoch in range(epochs):
    total_loss = 0
    for i in range(len(X)):
        output, hidden_states = forward_rnn(X[i].reshape(1, -1))
        backward_rnn(X[i].reshape(1, -1), y[i], output, hidden_states)
        total_loss += np.sum((output - y[i])**2)  # Mean Squared Error

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss / len(X)}")


# Predictions after training
predictions = []
for i in range(len(X)):
    output, _ = forward_rnn(X[i].reshape(1, -1))
    predictions.append(np.round(output).flatten()[0])

print("\n🔹 Actual Labels:", y.flatten())
print("🔹 Predicted Labels:", predictions)














