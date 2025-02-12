# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 15:38:59 2025

@author: srivi
"""

import numpy as np 
# a) Datenmatrix erstellen 

X = np.array([[0, 1, 1], 
              [1, 0, 1], 
              [0, 1, 1], 
              [1, 0, 1], 
              [0, 1, 1], 
              [1, 0, 1], 
              [0, 1, 1], 
              [1, 0, 1], 
              [0, 1, 1], 
              [1, 0, 1]]) 


# b) Zielmatrix definieren 

Y = np.array([[1, 0], 
              [0, 1], 
              [1, 0], 
              [0, 1], 
              [1, 0], 
              [0, 1], 
              [1, 0], 
              [0, 1], 
              [1, 0], 
              [0, 1]]) 

# c) Gewichtsmatrizen initialisieren 

W1 = np.random.rand(3, 3)  # Gewichte für die versteckte Schicht 
W2 = np.random.rand(3, 2)  # Gewichte für die Ausgabeschicht 
 
def sigmoid(x): 
    return 1 / (1 + np.exp(-x)) 
 
def sigmoid_derivative(x): 
    return x * (1 - x) 
 
# Trainingsprozess 
for epoch in range(1000): 

# d) Forward-Propagation 

    hidden_layer_input = np.dot(X, W1) 
    hidden_layer_output = sigmoid(hidden_layer_input) 
     
    output_layer_input = np.dot(hidden_layer_output, W2) 
    output = sigmoid(output_layer_input) 

# e) Backpropagation 
    error = Y - output 
    d_output = error * sigmoid_derivative(output) 
     
    error_hidden_layer = d_output.dot(W2.T) 
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output) 
     
    # Gewichte aktualisieren 
    W2 += hidden_layer_output.T.dot(d_output) 
    W1 += X.T.dot(d_hidden_layer) 

# f) Gesamtfehler berechnen 
total_error = np.sum(error**2).round(7)
 

# g) Aktualisierte Gewichte und finalen Gesamtfehler ausgeben 

print("Aktualisierte Gewichte für W1:\n", W1) 
print("Aktualisierte Gewichte für W2:\n", W2) 
print("Finaler Gesamtfehler:", total_error) 
