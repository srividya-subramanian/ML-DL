# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 15:18:30 2025

@author: srivi


You work as a data scientist and have the task of implementing a recurrent 
neural network (RNN) that is able to process simple sequences. The sequences 
consist of indices that correspond to words from a vocabulary. Your model should 
be able to predict the next word ID in the sequence. You should implement a 
cross entropy layer that compares your models predictions with the actual next 
word IDs and calculates the loss. Your model should have the ability to learn 
the word embeddings during training. 

a) Implement an embedding class that contains a forward method that takes a 
tensor of indices and returns the corresponding embeddings. Use a weight matrix 
that you initialize with random values. 

b) Implement a recurrent layer RNNCell consisting of three linear layers: one 
for the input, one for the hidden state and one for the output. The forward 
method of the class should receive the output of the previous hidden state and 
the input of the current training data and generate the new hidden state and 
the output. 


c) Implement a cross entropy layer CrossEntropyLoss that calculates the loss 
between the predictions of the model and the actual next word IDs. 

d) Write a training routine that trains the model with a simple data set. The 
data set consists of sequences of word IDs and the actual next word IDs as the 
target. Use a loop to iterate over the epochs and call the forward methods of 
the embedding and RNNCell classes as well as the CrossEntropyLoss method in 
each epoch. After each run, perform the backpropagation and update the weights
 with a simple SGD optimizer.  






"""