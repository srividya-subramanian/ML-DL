# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 06:14:05 2025

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

class LSTM:
    def __init__(self, input_dim, hidden_dim):
        self.hidden_dim = hidden_dim

        # Initialize weights
        self.W_f = np.random.randn(hidden_dim, input_dim + hidden_dim)  # Forget gate
        self.b_f = np.zeros((hidden_dim, 1))

        self.W_i = np.random.randn(hidden_dim, input_dim + hidden_dim)  # Input gate
        self.b_i = np.zeros((hidden_dim, 1))

        self.W_c = np.random.randn(hidden_dim, input_dim + hidden_dim)  # Cell state
        self.b_c = np.zeros((hidden_dim, 1))

        self.W_o = np.random.randn(hidden_dim, input_dim + hidden_dim)  # Output gate
        self.b_o = np.zeros((hidden_dim, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def forward(self, x, h_prev, c_prev):
        combined = np.vstack((h_prev, x))  # Concatenate previous hidden state and input
        outputs = []
        for idx in x:
            x_t = self.embedding[idx].reshape(-1, 1)

        # Compute gates
        f_t = self.sigmoid(np.dot(self.W_f, combined) + self.b_f)  # Forget gate
        i_t = self.sigmoid(np.dot(self.W_i, combined) + self.b_i)  # Input gate
        c_tilde = self.tanh(np.dot(self.W_c, combined) + self.b_c)  # Candidate cell state
        c_t = f_t * c_prev + i_t * c_tilde  # Update cell state
        o_t = self.sigmoid(np.dot(self.W_o, combined) + self.b_o)  # Output gate
        h_t = o_t * self.tanh(c_t)  # Compute new hidden state

        y_t = np.dot(self.W_y, h_t) + self.b_y
        outputs.append(y_t)
        return outputs, h_t, c_t

# Initialize LSTM
input_dim = 3  # Example input size
hidden_dim = 5  # Hidden layer size
lstm = LSTM(input_dim, hidden_dim)

# Sample input and previous states
x_t = np.random.randn(input_dim, 1)  # Input at time step t
h_prev = np.zeros((hidden_dim, 1))  # Previous hidden state
c_prev = np.zeros((hidden_dim, 1))  # Previous cell state

# Perform forward pass
h_next, c_next = lstm.forward(x_t, h_prev, c_prev)
print("Next Hidden State:\n", h_next)
print("Next Cell State:\n", c_next)












