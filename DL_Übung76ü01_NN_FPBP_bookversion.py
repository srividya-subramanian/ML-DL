# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 06:23:33 2025

@author: srivi
"""

import numpy as np 

 

# Sigmoid-Funktion 

def sigmoid(x): 
    return 1 / (1 + np.exp(-x)) 
 

# Ableitung der Sigmoid-Funktion 

def sigmoid_derivative(x): 
    return x * (1 - x) 
 

# Datensatz 

X = np.array([[5.1, 3.5, 1.4, 0.2], 
              [7.0, 3.2, 4.7, 1.4], 
              [4.6, 3.1, 1.5, 0.2], 
              [6.5, 2.8, 4.6, 1.5], 
              [5.0, 3.6, 1.4, 0.2], 
              [5.7, 2.8, 4.5, 1.3]]) 
y = np.array([[0], [1], [0], [1], [0], [1]]) 
 

# Initialisierung der Gewichte und Biases 

np.random.seed(42) 
weights_input_hidden = np.random.rand(4, 32) 
weights_hidden_output = np.random.rand(32, 1) 
bias_hidden = np.random.rand(1, 32) 
bias_output = np.random.rand(1, 1) 
 
print(weights_input_hidden.shape, weights_hidden_output.shape,bias_hidden.shape,bias_output.shape)
# Trainingsphase 

for i in range(10000): 
    # Vorwärtspropagierung 

    hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden 
    hidden_layer_output = sigmoid(hidden_layer_input) 
 
    final_output_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output 
    final_output = sigmoid(final_output_input) 

 

    # Berechnung des Fehlers 

    error = y - final_output 
    d_predicted_output = error * sigmoid_derivative(final_output) 
 

    # Backpropagation 

    error_hidden_layer = d_predicted_output.dot(weights_hidden_output.T) 
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output) 
 

    # Aktualisierung der Gewichte und Biases 

    weights_hidden_output += hidden_layer_output.T.dot(d_predicted_output) 
    weights_input_hidden += X.T.dot(d_hidden_layer) 
    bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) 
    bias_output += np.sum(d_predicted_output, axis=0, keepdims=True) 
 

# Ausgabe der aktualisierten Gewichte und Biases 

print("Aktualisierte Gewichte von Eingabe zu versteckter Schicht:") 
print(weights_input_hidden) 
print("Aktualisierte Gewichte von versteckter Schicht zu Ausgabe:") 
print(weights_hidden_output) 
print("Aktualisierte Biases der versteckten Schicht:") 
print(bias_hidden) 
print("Aktualisierter Bias der Ausgabeschicht:") 
print(bias_output) 
print((y, final_output))
