# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 05:51:25 2025

@author: srivi
"""

import numpy as np

class SimpleLSTM:
    def __init__(self, isize, hsize, osize):
        self.isize = isize #input size
        self.hsize = hsize #hidden size
        self.osize = osize #output size

        # Xavier Initialization for weights and zeros for biases
        self.Wf = np.random.randn(hsize, hsize + isize) * 0.01  # Forget gate
        self.Wi = np.random.randn(hsize, hsize + isize) * 0.01  # Input gate
        self.Wo = np.random.randn(hsize, hsize + isize) * 0.01  # Output gate
        self.Wc = np.random.randn(hsize, hsize + isize) * 0.01  # Candidate memory

        self.bf = np.zeros((hsize, 1))  # Forget bias
        self.bi = np.zeros((hsize, 1))  # Input bias
        self.bo = np.zeros((hsize, 1))  # Output bias
        self.bc = np.zeros((hsize, 1))  # Candidate memory bias

        # Output layer weights and bias
        self.Wy = np.random.randn(osize, hsize) * 0.01
        self.by = np.zeros((osize, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def forward(self, X):
        """
        Forward pass for an LSTM on a batch of sequences.
        X: (sequence_length, isize) -> input sequence
        """
        sequence_length, _ = X.shape
        h_t = np.zeros((self.hsize, 1))  # Initial hidden state
        c_t = np.zeros((self.hsize, 1))  # Initial cell state

        for t in range(sequence_length):
            x_t = X[t].reshape(-1, 1)  # Get current timestep input
            
            # Concatenate hidden state and input
            concat = np.vstack((h_t, x_t))

            # Compute LSTM gate activations
            f_t = self.sigmoid(np.dot(self.Wf, concat) + self.bf)  # Forget gate
            i_t = self.sigmoid(np.dot(self.Wi, concat) + self.bi)  # Input gate
            o_t = self.sigmoid(np.dot(self.Wo, concat) + self.bo)  # Output gate
            c_hat_t = self.tanh(np.dot(self.Wc, concat) + self.bc)  # Candidate memory
            
            # Compute new cell state and hidden state
            c_t = f_t * c_t + i_t * c_hat_t
            h_t = o_t * self.tanh(c_t)

        # Output layer (linear transformation)
        y = np.dot(self.Wy, h_t) + self.by

        return y, h_t

# Example usage
isize = 3    # Features per time step
hsize = 5   # LSTM hidden units
osize = 1   # Output dimension

lstm = SimpleLSTM(isize, hsize, osize)

# Dummy input sequence of length 4, with 3 features each
X = np.random.randn(4, isize)  

output, hidden_state = lstm.forward(X)

print("LSTM Output:", output.flatten())











