# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 05:54:58 2025

@author: srivi


Develop a simple character-based RNN for language modeling without the use of 
LSTM cells. Your task is to create a model that learns to predict the next 
character in a sequence based on the previous characters. Use truncated 
backpropagation to mitigate the problems with vanishing and exploding gradients.
You should perform the following steps: 

a) Create an artificial alphabet of 15 characters and generate a sequence of 
    1000 characters by randomly selecting characters from your alphabet.  

b) Implement an RNN in Python that contains an embedding layer and a simple RNN 
    cell. Do not use LSTM cells. Set the dimension of the embedding layer to 10 
    and the number of hidden neurons to 20. 

c) Train the RNN with your generated sequence. Use a batch size of 32 and a 
    sequence length (bptt) of 5. Perform the backpropagation after each sequence 
    and update the weights. 

d) Write a function that calculates the loss function (cross-entropy loss) and 
    another function that performs the optimization step (e.g. with SGD). 

e) Run the training for 100 iterations and save the loss after each iteration
    to monitor the performance of the model.  
  
"""
import numpy as np
import numpy.random as rand
import random

alphabet = list("ABCDEFGHIJKLMNO")
random_sequence = ''.join(random.choices(alphabet, k=1000))
#print(random_sequence)

alphabet_index = {ch: i for i, ch in enumerate(alphabet)}
indexed_seq = np.array([alphabet_index[ch] for ch in random_sequence])

input_size = 10 #dimension of the embedding layer to 10 
hidden_size = 20 
output_size = 1
seq_length = 1000
vocab_size = 15

class SimpleRNN:
    def __init__(self, vocab_size, input_size, hidden_size, output_size):
        self.hsize = hidden_size
        self.isize = input_size
        self.osize = output_size
        
        # Initialize embedding layer
        self.embedding = np.random.randn(vocab_size, input_size) * 0.01
        
        # Initialize weights
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01  # Input to hidden
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # Hidden to hidden
        self.Why = np.random.randn(output_size, hidden_size) * 0.01  # Hidden to output
        
        # Initialize bias
        self.bh = np.zeros(hidden_size)
        self.by = np.zeros(output_size)
        
        # Hidden state initialization
        self.h = np.zeros((hidden_size, 1))

    def forward(self, x):
        hx = np.dot(self.Wxh, x) + np.dot(self.Whh, self.h)  # Hidden state computation
        ho  = np.tanh(hx)
        self.h = ho  # Activation
        y = np.dot(self.Why, self.h)  # Output
        return y, self.h

def create_batches(indexed_sequence, batch_size=32, seq_len=5):
    
    X_batches = []
    Y_batches = []
    num_batches = len(indexed_seq) // (batch_size * seq_len)

    for i in range(num_batches):
        start_idx = i * batch_size * seq_len
        batch_X = []
        batch_Y = []
        
        for j in range(batch_size):
            start_seq = start_idx + j * seq_len
            batch_X.append(indexed_sequence[start_seq:start_seq + seq_len])
            batch_Y.append(indexed_sequence[start_seq + 1:start_seq + seq_len + 1])  # Next character
        
        X_batches.append(np.array(batch_X))
        Y_batches.append(np.array(batch_Y))
    
    return X_batches, Y_batches

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Stability trick
    y = exp_x / exp_x.sum(axis=1, keepdims=True)
    return y

def cross_entropy_loss(y_pred, y_true):
    m = y_pred.shape[1]
    probs = softmax(y_pred)  # Convert logits to probabilities
    log_probs = -np.log(probs[y_true, np.arange(m)])
    loss= np.mean(log_probs)
    return loss

def sgd_update(model, dWx, dWh, dWy, dbh, dby, lr=0.01):
    """
    Updates model parameters using Stochastic Gradient Descent (SGD).
    """
    model.Wxh -= lr * dWx
    model.Whh -= lr * dWh
    model.Why -= lr * dWy
    model.bh -= lr * dbh
    model.by -= lr * dby


X_batches, Y_batches = create_batches(indexed_seq, batch_size=32, seq_len=5)

rnn = SimpleRNN(vocab_size, input_size, hidden_size, output_size)
epochs = 100
lr = 0.01 # learning_rate
batch_size = 32
seq_len = 5
hidden_size = rnn.h

losses = []
for epoch in range(epochs):
    batch_loss = 0
    total_loss = 0
    for X_batch, Y_batch in zip(X_batches, Y_batches):
        dWx, dWh, dWy = np.zeros_like(rnn.Wxh), np.zeros_like(rnn.Whh), np.zeros_like(rnn.Why)
        dbh, dby = np.zeros_like(rnn.bh), np.zeros_like(rnn.by)
        for i in range(batch_size):
            input, target  = X_batches[i], Y_batches[i]
            
            # Forward pass
            outputs, h = rnn.forward(input)
            y_pred = np.hstack(outputs)
            batch_loss += cross_entropy_loss(y_pred, target)
            
            # Backpropagation through time (BPTT)
            dWy += np.dot((softmax(y_pred) - np.eye(15)[target].T), h[-1].T)
            dby += np.sum(softmax(y_pred) - np.eye(15)[target].T, axis=1, keepdims=True)
        
        # Normalize gradients
        dWy /= batch_size
        dby /= batch_size
        
        # Perform SGD step
        rnn.sgd_update(dWx, dWh, dWy, dbh, dby, lr)
        
        total_loss += batch_loss / batch_size
    
    losses.append(total_loss / len(X_batches))
    print(f"Epoch {epoch+1}/{epochs}, Loss: {losses[-1]:.4f}")



import matplotlib.pyplot as plt

plt.plot(losses, label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss over Time")
plt.legend()
plt.show()




