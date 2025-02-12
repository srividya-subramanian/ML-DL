# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 15:57:26 2025

@author: srivi
"""
import numpy as np
input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) 
target = np.array([[0], [1], [1], [0]]) 


np.random.seed(42)  # Für reproduzierbare Ergebnisse 
w = np.random.rand(2, 1)  # Zwei Gewichte für zwei Eingangssignale 
alpha = 0.1 
epoch = 100

def perceptron_npdot(x, w): 
    return np.dot(x, w)
def sigmoid(x): 
    return 1 / (1 + np.exp(-x))  
def error(y, y_pred):
    return (y - y_pred)** 2
def delta(y, y_pred):
    return (y - y_pred)

for i in range(epoch):
    terr=0
    for j in range(len(target)):
        x = input[j]
        y = target[j]
        y_pred = perceptron_npdot(x, w)
        err = error(y, y_pred)
        terr +=  err
        deltay = delta(y, y_pred)
        w += x.reshape(-1, 1) * deltay * alpha
    print(terr)
    
for i in range(len(target)): 
    prediction = input[i].dot(w)     
    print(target[i], prediction)
    
    