# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 15:14:36 2025

@author: srivi

You work as a data scientist in a company that specializes in the development 
of language models. Your current project deals with the implementation of an 
LSTM network for character-based language modeling. The goal is to train a model 
that is able to predict the next character based on a given sequence of characters. 
Your task is to design the architecture of the LSTM network and implement an 
abbreviated backpropagation. 

a) Design an LSTM architecture consisting of an embedding layer, an LSTM layer 
and an output layer. The embedding layer should convert the characters into a 
128-dimensional vector. The LSTM layer should have 256 hidden units and the 
output layer should predict the probabilities for the next character. Name all 
layers and parameters of your model. 

b) Implement the initialization of the model in Python without using the 
Tensorflow library. However, you may use the numpy library for the vector and 
matrix operations. Explain each step of your code and the function of the 
parameters used. 

c) use a data set of randomly generated strings consisting of the letters 
A to Z. The data set should contain 1000 strings, with each string being 50 
characters long. Save this data set in a table. 


d) Describe how you would implement truncated backpropagation in your model.
Indicate how many time steps you would use for the backpropagation and justify 
your choice.  

🔹 Plan
1. Design the LSTM architecture
    * An embedding layer (128-dim vector per character)
    * An LSTM layer (256 hidden units)
    * An output layer (predicts next character probabilities)
2. Implement model initialization using NumPy
3. Generate a synthetic dataset of 1000 sequences (50 characters each)
4. Explain truncated backpropagation through time (BPTT)


📍 Model Layers & Parameters
Layer	                Description
Embedding Layer	        Converts characters into 128-dimensional vectors.
LSTM Layer	            Has 256 hidden units and computes hidden & cell states.
Output Layer	        Fully connected layer that predicts the next character.

📍 Model Parameters
Parameter	      Shape	        Description
W_embed	          (26, 128)	    Embedding weights (one for each character A-Z).
Wf, Wi, Wo, Wc	  (256, 128)	LSTM weight matrices for forget, input, output, and cell gates.
Uf, Ui, Uo, Uc	  (256, 256)	LSTM recurrent weight matrices.
bf, bi, bo, bc	  (256, 1)	    Bias terms for LSTM gates.
Wy	              (26, 256)	    Fully connected layer for character prediction.
by	              (26, 1)	    Bias for output layer.



"""
import numpy as np

class LSTMModel:
    def __init__(self, vocab_size=26, embedding_dim=128, hidden_size=256, output_size=26):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Embedding layer
        self.embedding = np.random.randn(vocab_size, embedding_dim) * 0.01

        # LSTM Weights
        self.W_i = np.random.randn(hidden_size, embedding_dim) * 0.01  # Input gate
        self.W_f = np.random.randn(hidden_size, embedding_dim) * 0.01  # Forget gate
        self.W_o = np.random.randn(hidden_size, embedding_dim) * 0.01  # Output gate
        self.W_c = np.random.randn(hidden_size, embedding_dim) * 0.01  # Candidate memory

        self.U_i = np.random.randn(hidden_size, hidden_size) * 0.01
        self.U_f = np.random.randn(hidden_size, hidden_size) * 0.01
        self.U_o = np.random.randn(hidden_size, hidden_size) * 0.01
        self.U_c = np.random.randn(hidden_size, hidden_size) * 0.01

        self.b_i = np.zeros((hidden_size, 1))
        self.b_f = np.zeros((hidden_size, 1))
        self.b_o = np.zeros((hidden_size, 1))
        self.b_c = np.zeros((hidden_size, 1))

        # Output layer
        self.W_y = np.random.randn(output_size, hidden_size) * 0.01
        self.b_y = np.zeros((output_size, 1))

    def forward(self, inputs, h_prev, c_prev):
        """
        Forward pass through LSTM.
        """
        h_t, c_t = h_prev, c_prev
        outputs = []

        for idx in inputs:
            x_t = self.embedding[idx].reshape(-1, 1)

            # Compute LSTM gate activations
            f_t = self.sigmoid(np.dot(self.W_f, x_t) + np.dot(self.U_f, h_t) + self.b_f)
            i_t = self.sigmoid(np.dot(self.W_i, x_t) + np.dot(self.U_i, h_t) + self.b_i)
            c_tilde = np.tanh(np.dot(self.W_c, x_t) + np.dot(self.U_c, h_t) + self.b_c)

            o_t = self.sigmoid(np.dot(self.W_o, x_t) + np.dot(self.U_o, h_t) + self.b_o)

            # Update cell state and hidden state
            c_t = f_t * c_t + i_t * c_tilde
            h_t = o_t * np.tanh(c_t)

            # Compute output logits
            y_t = np.dot(self.W_y, h_t) + self.b_y
            outputs.append(y_t)

        return outputs, h_t, c_t

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

# Initialize the model
lstm_model = LSTMModel()        
        
        
import pandas as pd
import random
import string

# Generate 1000 random strings of length 50
num_strings = 1000
string_length = 50

data = [''.join(random.choices(string.ascii_uppercase, k=string_length)) for _ in range(num_strings)]

# Save to a Pandas DataFrame
df = pd.DataFrame(data, columns=["Random_String"])

# Save to CSV
df.to_csv("random_strings.csv", index=False)

print(df.head())  # Show first few entries


'''
📌 Part (d): Truncated Backpropagation Through Time (BPTT)
📝 What is BPTT?
Standard backpropagation is inefficient for sequences (memory issues).
Truncated BPTT splits training into small time steps to optimize performance.

🔹 Steps for Truncated BPTT
    * Split each sequence into chunks of 10 time steps.
    * Compute gradients only within these 10 steps.
    * Pass hidden states between chunks but stop gradients to earlier steps.

truncated_time_steps = 10  # Number of time steps per BPTT pass

🔍 Why 10 Time Steps?
✔ Balances efficiency & learning (shorter reduces context, longer slows training).
✔ Reduces memory usage vs. full BPTT.
✔ Works well in practical LSTM training.


'''

        
def truncated_bptt(model, inputs, targets, time_steps=10, lr=0.01):
    h_t = np.zeros((model.hidden_size, 1))
    c_t = np.zeros((model.hidden_size, 1))

    total_loss = 0

    for i in range(0, len(inputs) - time_steps, time_steps):
        x_batch = inputs[i:i+time_steps]
        y_batch = targets[i:i+time_steps]

        outputs, h_t, c_t = model.forward(x_batch, h_t, c_t)
        
        # Compute loss (cross-entropy)
        loss = -np.sum(np.log(np.exp(outputs[-1]) / np.sum(np.exp(outputs[-1]), axis=0)))
        total_loss += loss

        # Backpropagate over last 10 time steps (to be implemented)
        # Compute gradients and update weights here...

    return total_loss

# Example usage
loss = truncated_bptt(lstm_model, np.random.randint(0, 26, size=50), np.random.randint(0, 26, size=50))
print("Loss:", loss)
        
        
        
        
        
        
        
    