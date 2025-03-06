# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 14:31:18 2025

@author: srivi
"""

import numpy as np 
import random 
 

# Alphabet und Sequenz generieren 

alphabet = 'abcdefghijklmno' 
sequence = ''.join(random.choices(alphabet, k=1000)) 
 

# Zeichen in Indizes umwandeln 

char_to_index = {ch: i for i, ch in enumerate(alphabet)} 
index_to_char = {i: ch for i, ch in enumerate(alphabet)} 
indices = np.array([char_to_index[ch] for ch in sequence]) 
 

# RNN-Modell definieren 

def init_weights(input_dim, hidden_dim, output_dim): 
    return (np.random.randn(input_dim, hidden_dim),  
            np.random.randn(hidden_dim, hidden_dim),  
            np.random.randn(hidden_dim, output_dim)) 
 
def init_bias(hidden_dim, output_dim): 
    return (np.zeros(hidden_dim), np.zeros(output_dim)) 
 
def softmax(x): 
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))  
    return e_x / e_x.sum(axis=1, keepdims=True)   
class RNN: 
    def __init__(self, input_size, hidden_size, output_size): 
        self.Wxh, self.Whh, self.Why = init_weights(input_size, hidden_size, output_size) 
        self.bh, self.by = init_bias(hidden_size, output_size) 
        self.hidden = np.zeros(hidden_size) 
        print(self.bh)
    def forward(self, inputs): 
        hs, ys, ps = {}, {}, {} 
        hs[-1] = np.copy(self.hidden) 
        for t in range(len(inputs)): 
#            self.hidden = np.tanh(np.dot(self.Wxh.T, inputs[t].T) + np.dot(self.Whh.T, self.hidden.reshape(-1, 1))) #+ self.bh) 
            self.hidden = np.tanh(np.dot(self.Wxh, inputs[t]) + np.dot(self.Whh, self.hidden) + self.bh.reshape(-1, 1))            
            y = np.dot(self.Why.T, self.hidden)
            p = softmax(y) 
            hs[t] = self.hidden 
            ys[t] = y 
            ps[t] = p 
        return hs, ys, ps 
 
    def backprop(self, inputs, targets, hs, ps): 
        dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why) 
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by) 
        next_hidden_grad = np.zeros_like(self.hidden) 
        loss = 0 
        for t in reversed(range(len(inputs))): 
            dy = np.copy(ps[t]) 
            dy -= np.eye(len(alphabet))[targets[t]].T             
            dWhy += np.dot(dy, hs[t].T) 
            dby += dy 
            dh = np.dot(self.Why.T, dy) + next_hidden_grad 
            dh_raw = (1 - hs[t] * hs[t]) * dh 
            dbh += dh_raw 
            dWxh += np.outer(dh_raw, inputs[t]) 
            dWhh += np.outer(dh_raw, hs[t-1]) 
            next_hidden_grad = np.dot(self.Whh.T, dh_raw) 
            loss += -np.log(ps[t][targets[t]]) 
        return loss, dWxh, dWhh, dWhy, dbh, dby 
 
    def update_weights(self, dWxh, dWhh, dWhy, dbh, dby, lr): 
        self.Wxh -= lr * dWxh 
        self.Whh -= lr * dWhh 
        self.Why -= lr * dWhy 
        self.bh -= lr * dbh 
        self.by -= lr * dby 
 






# Training 

def train_rnn(rnn, data, bptt, batch_size, epochs, lr): 
    n_batches = len(data) // batch_size 
    trimmed_data = data[:n_batches * batch_size] 
    batched_data = trimmed_data.reshape(batch_size, n_batches) 
    batched_data = batched_data.transpose() 
     
    for epoch in range(epochs): 
        total_loss = 0 
        for batch_i in range(0, n_batches, bptt): 
            hidden = np.zeros(rnn.hidden.shape) 
            batch_loss = 0 
            for t in range(bptt): 
                inputs = np.eye(len(alphabet))[batched_data[batch_i:batch_i+bptt].T] 
                targets = batched_data[batch_i+1:batch_i+bptt+1].T.flatten()
                hs, _, ps = rnn.forward(inputs) 
                loss, dWxh, dWhh, dWhy, dbh, dby = rnn.backprop(inputs, targets, hs, ps) 
                rnn.update_weights(dWxh, dWhh, dWhy, dbh, dby, lr) 
                batch_loss += loss 
            total_loss += batch_loss 
        print(f'Epoch {epoch}, Loss: {total_loss}') 
 

# Initialisiere RNN 

rnn = RNN(input_size=len(alphabet), hidden_size=20, output_size=len(alphabet)) 
 

# Trainiere das RNN 

train_rnn(rnn, indices, bptt=5, batch_size=32, epochs=100, lr=0.1) 

