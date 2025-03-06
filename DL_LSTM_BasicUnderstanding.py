# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 04:59:00 2025

@author: srivi
🔹 Long Short-Term Memory (LSTM) Networks
LSTM (Long Short-Term Memory) is a special type of Recurrent Neural Network (RNN) 
designed to handle long-term dependencies and sequential data more effectively 
than standard RNNs. It overcomes the vanishing gradient problem, allowing networks
to remember information over longer sequences.

🔹 Why Use LSTMs?
🚀 Captures long-range dependencies – Retains memory of past inputs for a long time
🚀 Prevents vanishing gradient problem – Efficiently trains deep networks
🚀 Ideal for sequential tasks – Used in NLP, speech recognition, time series forecasting

🔹 LSTM vs. Standard RNN
Feature	Standard            RNN	                    LSTM
Memory Retention	       Short-term only	    Long-term (with memory cells)
Vanishing Gradient	     Severe issue	        Mitigated
Suitable for	       Short sequences	        Long sequences
Example Use Case	Simple text generation	    Sentiment analysis, translation

🔹 LSTM Architecture
Unlike a simple RNN that only has a single activation function, an LSTM unit has three gates:

Forget Gate – Decides what information to discard from memory.
Input Gate – Determines which new information to store.
Output Gate – Decides the final output of the cell.

🔷 Cell State (C_t) stores long-term memory, modified by the gates.
🔷 Hidden State (h_t) carries information to the next timestep.

🔹 LSTM Implementation Without TensorFlow/Keras (NumPy Only)
We implement a simple LSTM forward pass using NumPy.

🔹 Applications of LSTMs
✅ Speech Recognition – Converts speech into text
✅ Machine Translation – Translates one language to another
✅ Stock Market Prediction – Forecasts financial trends
✅ Chatbots – Used in conversational AI
"""

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))



vocab_size=26
embedding_dim=128
hidden_size=256
output_size=26
vocab_size = vocab_size
embedding_dim = embedding_dim
hidden_size = hidden_size
output_size = output_size

# Embedding layer
embedding = np.random.randn(vocab_size, embedding_dim) * 0.01

# LSTM Weights
W_i = np.random.randn(hidden_size, embedding_dim) * 0.01  # Input gate
W_f = np.random.randn(hidden_size, embedding_dim) * 0.01  # Forget gate
W_o = np.random.randn(hidden_size, embedding_dim) * 0.01  # Output gate
W_c = np.random.randn(hidden_size, embedding_dim) * 0.01  # Candidate memory

U_i = np.random.randn(hidden_size, hidden_size) * 0.01
U_f = np.random.randn(hidden_size, hidden_size) * 0.01
U_o = np.random.randn(hidden_size, hidden_size) * 0.01
U_c = np.random.randn(hidden_size, hidden_size) * 0.01

b_i = np.zeros((hidden_size, 1))
b_f = np.zeros((hidden_size, 1))
b_o = np.zeros((hidden_size, 1))
b_c = np.zeros((hidden_size, 1))

inputs = np.random.randint(0, 26, size=50)
idx = inputs[0]
x_t = embedding[idx].reshape(-1, 1)

# Output layer
W_y = np.random.randn(output_size, hidden_size) * 0.01
b_y = np.zeros((output_size, 1))



h_t = np.zeros((hidden_size, 1))
c_t = np.zeros((hidden_size, 1))

# Compute LSTM gate activations
f_t = sigmoid(np.dot(W_f, x_t) + np.dot(U_f, h_t) + b_f)
i_t = sigmoid(np.dot(W_i, x_t) + np.dot(U_i, h_t) + b_i)
c_tilde = np.tanh(np.dot(W_c, x_t) + np.dot(U_c, h_t) + b_c)

o_t = sigmoid(np.dot(W_o, x_t) + np.dot(U_o, h_t) + b_o)

# Update cell state and hidden state
c_t = f_t * c_t + i_t * c_tilde
h_t = o_t * np.tanh(c_t)

# Compute output logits
y_t = np.dot(W_y, h_t) + b_y
# outputs.append(y_t)

