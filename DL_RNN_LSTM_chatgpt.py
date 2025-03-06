# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 08:53:44 2025

@author: srivi

🔹 Difference Between RNN and LSTM
Feature	            RNN (Recurrent Neural Network)	         LSTM (Long Short-Term Memory)
Memory Handling	    Short-term memory only	                 Long-term and short-term memory
Vanishing Gradient	Suffers from vanishing gradient problem	 Uses gates to solve vanishing gradient issue
Structure	        Simple recurrent connections	         Includes Forget, Input, and Output gates
Performance on 	    Poor (loses context over long 	         Good (remembers context over long sequences)
Long Sequences      sequences)
Use Cases	        Simple sequential tasks like 	         Complex tasks like speech recognition, 
                    next-word prediction                     language modeling
Training Stability	Harder to train due to exploding/	     More stable training due to controlled
                    vanishing gradients                      memory updates
"""
import numpy as np

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

'''
def forward_propagation(x,wi,wo,bi,bo):
    #print(x.shape,wi.shape)
    #print(((perceptron_npdot(x,wi))+bi).shape)
    hx = perceptron_npdot(x,wi)+bi
    ho  = relu_activationfn(hx)
    yh = perceptron_npdot(ho,wo)+bo
    y_pred = sigmoid_activationfn(yh)
    return y_pred, ho
'''

class SimpleLSTM:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        
        # Initialize weights
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size) * 0.01  # Forget gate
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size) * 0.01  # Input gate
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size) * 0.01  # Output gate
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size) * 0.01  # Cell state
        
        self.Why = np.random.randn(output_size, hidden_size) * 0.01  # Output layer
        
        # Hidden state & cell state initialization
        self.h = np.zeros((hidden_size, 1))
        self.c = np.zeros((hidden_size, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        combined = np.vstack((self.h, x))  # Combine previous hidden state and input
        
        forget_gate = self.sigmoid(np.dot(self.Wf, combined))
        input_gate = self.sigmoid(np.dot(self.Wi, combined))
        output_gate = self.sigmoid(np.dot(self.Wo, combined))
        cell_candidate = np.tanh(np.dot(self.Wc, combined))
        
        self.c = forget_gate * self.c + input_gate * cell_candidate  # Update cell state
        self.h = output_gate * np.tanh(self.c)  # Update hidden state
        
        y = np.dot(self.Why, self.h)  # Output
        return y, self.h
    
# Input: One-hot vector (5 words in vocab)
x = np.array([[0], [1], [0], [0], [0]])  # Represents "blue"

rnn = SimpleRNN(input_size=5, hidden_size=10, output_size=5)
lstm = SimpleLSTM(input_size=5, hidden_size=10, output_size=5)

rnn_output, rnn_hidden = rnn.forward(x)
lstm_output, lstm_hidden = lstm.forward(x)

print("RNN Output:\n", rnn_output)
print("\nLSTM Output:\n", lstm_output)
    