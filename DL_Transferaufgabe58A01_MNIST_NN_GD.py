# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 13:55:38 2025

@author: srivi
"""

import numpy as np
import pandas as pd

def ReLU(Z):
    return np.maximum(Z, 0)

def forward_pass(x,w,b):
    pred = np.dot(x,w)+b
    a1 = ReLU(pred)
    return pred

def calculate_error(y, y_pred):
    err = (y_pred - y)
    return err

def update_weights(x, w, y, y_pred, lr,epoch): 
    for i in range(epoch):
        err = calculate_error(y, y_pred)
        w += (np.dot(x.T,err) * lr)
    print(w)    
    return w

# Initialize weights and bais
import numpy.random as rnd
def init_params():
    lr = 0.01
    w = rnd.rand(784,10) - 0.5
    b = rnd.rand(1,10) - 0.5
    return w, b, lr

w, b, lr = init_params()

train = pd.read_csv('/Users/srivi/Documents/ML_data/mnist_train.csv')
test = pd.read_csv('/Users/srivi/Documents/ML_data/mnist_test.csv')

y_train = np.eye(10)[train['label']]
y_test = np.eye(10)[test['label']]
X_train = train.drop('label', axis=1)/ 255.0 
X_test = test.drop('label', axis=1)/ 255.0 


y_pred = forward_pass(X_train,w, b)
#    print(y_train-y_pred)
w = update_weights(X_train, w, y_train, y_pred, lr, 100)

import matplotlib.pyplot as plt 
def visualize_weights(weights, neuron_idx): 
    neuron_weights = weights[:, neuron_idx].reshape(28, 28) 
    plt.imshow(neuron_weights, cmap='hot', interpolation='nearest') 
    plt.colorbar() 
    plt.title(f'Gewichte für Neuron {neuron_idx}') 
    plt.show() 
 
for i in range(0,10):
    visualize_weights(w, i)  # Neuron für die Ziffer "2" 
 
