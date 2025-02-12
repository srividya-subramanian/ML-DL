# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 16:13:40 2025

@author: srivi
"""

import numpy as np 
import matplotlib.pyplot as plt 
from tensorflow.keras.datasets import mnist 
# a) Laden der Daten 

(train_images, train_labels), _ = mnist.load_data() 
images = train_images[:1000].reshape(1000, 784) / 255.0 
labels = np.eye(10)[train_labels[:1000]] 

# b) Initialisierung der Gewichte und Definition der Forward- und Backward-Propagation 

weights = np.random.rand(784, 10) * 0.01 
def softmax(x): 
    e_x = np.exp(x - np.max(x)) 
    return e_x /e_x.sum(axis=1, keepdims=True) 
 
def predict(images, weights): 
    return softmax(np.dot(images, weights)) 
 
def train(images, labels, weights, epochs, lr): 
    for epoch in range(epochs): 
        predictions = predict(images, weights) 
        error = predictions - labels 
        weights -= lr * np.dot(images.T, error) 
        break
    return weights 




# c) Training des Netzwerks 

epochs = 100 
learning_rate = 0.1 
trained_weights = train(images, labels, weights, epochs, learning_rate) 


# d) Visualisierung der Gewichte eines Ausgabeneurons 

def visualize_weights(weights, neuron_idx): 
    neuron_weights = weights[:, neuron_idx].reshape(28, 28) 
    plt.imshow(neuron_weights, cmap='hot', interpolation='nearest') 
    plt.colorbar() 
    plt.title(f'Gewichte für Neuron {neuron_idx}') 
    plt.show() 
 
visualize_weights(trained_weights, 2)  # Neuron für die Ziffer "2" 
 