# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 15:24:43 2025

@author: sriv

This implementation includes:

(a) Embedding Layer: Converts word indices into dense vectors.
(b) RNNCell: Computes hidden states and outputs using linear layers.
(c) CrossEntropyLoss: Computes the loss between predictions and targets.
(d) Training Routine: Loops over the dataset, performs forward and backward
 passes, and updates weights using SGD.

"""
import numpy as np

# Build vocabulary
# vocabulary of five words related to sky, as the test sample is “sky blue clouds”
vocab = ["blue", "cloudy", "sunny", "rainy", "sky"]
vocab_index = {word: i for i, word in enumerate(sorted(vocab))}  # Assign index

# ✅ Simple Dataset
vocab_size = 5  # Vocabulary of 5 words (IDs: 0-4)
input_size = 4  # Each word maps to a 4D vector
hidden_size = 6  # RNN hidden state size
output_size = vocab_size  # Output size same as vocab size

class Embedding:
    def __init__(self, vocab_size, embedding_dim):
         self.weights = np.random.randn(vocab_size, embedding_dim) * 0.1  # Random initialization

    def forward(self, indices):
        """Retrieve embeddings for the given word indices."""
        return self.weights[indices]  # Lookup table

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        
        # Initialize weights
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01  # Input to hidden
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # Hidden to hidden
        self.Why = np.random.randn(output_size, hidden_size) * 0.01  # Hidden to output
        
        # Hidden state initialization
        self.h = np.zeros((hidden_size, 1))

    def forward(self, x):
        self.h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, self.h))  # Activation
        y = np.dot(self.Why, self.h)  # Output
        return y, self.h




sequences = [("rainy","day"),
             ("cloudy","sky"),
             ("wet","weather"),
             ("jacket","needed")]

np.array([0, 1, 2, 3])  # Word IDs in a sequence
targets = np.array([1, 2, 3, 4])  # Next word IDs
