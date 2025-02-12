# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 10:54:09 2025

@author: srivi
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 12:58:39 2024

@author: srivi
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.random as rand
from math import e 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def numbers(): 
    array = np.array([
       (([[0, 1, 0], [1, 1, 1], [0, 1, 0], [1, 1, 1], [0, 1, 0]]),0),  # Ziffer 0 
       ([[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]],1),  # Ziffer 1 
       ([[0, 1, 1], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1]],2),  # Ziffer 2 
       ([[1, 1, 1], [0, 0, 1], [0, 1, 1], [0, 0, 1], [1, 1, 1]],3),  # Ziffer 3 
       ([[1, 0, 0], [1, 1, 0], [1, 1, 1], [0, 1, 0], [0, 1, 0]],4),  # Ziffer 4 
       ([[1, 1, 1], [1, 0, 0], [1, 1, 1], [0, 0, 1], [1, 1, 1]],5),   # Ziffer 5 
       ([[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]],6),  # Ziffer 1 
       ([[0, 1, 1], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1]],7),  # Ziffer 2 
       ([[1, 1, 1], [0, 0, 1], [0, 1, 1], [0, 0, 1], [1, 1, 1]],8),  # Ziffer 3 
       ([[1, 0, 0], [1, 1, 0], [1, 1, 1], [0, 1, 0], [0, 1, 0]],9)  # Ziffer 4 
    ])
    return array


def perceptron(x, w): 
    y = np.dot(x, w)
    return y

def activationfn(y):
    sig = 1 / (1+e**(-y))
    return sig

def derivativefn(y):
    dsig = e**(-y)/ (1+e**(-y))
    return dsig

def forward_propagation(x,wi,wo):
    hx = perceptron(x,wi)
    h  = activationfn(hx)
    hy = perceptron(h,wo)
    y_pred = activationfn(hy)
    return y_pred, h

def error(y, y_pred):
    err = y - y_pred
    return err

def back_propagation(h, wi, wo, x, y, y_pred,  lr):
    err = error(y, y_pred)
    dy_pred = err * derivativefn(y_pred)
    errh = np.dot(dy_pred, wo.T)
    dh = errh * derivativefn(h)
    
    wi += np.dot(x.T, dh)*lr
    wo += np.dot(h.T,y_pred)*lr
    return wi, wo

inputLayerSize, hiddenLayerSize, outputLayerSize = 15, 10, 5
wi = np.random.uniform(0.01,0.3,size=(15,10))
wo = np.random.uniform(0.01,0.3,size=(10,5)) 
print(wi, wo)

train_inputs = numbers()
train_outputs = np.array([[1], [2], [3], [4], [5],[1], [2], [3], [4], [5],[1], [2], [3], [4], [5]]) 

lr = 0.7
epochs = 100000
for epoch in range(epochs): 
   y_pred, h = forward_propagation(train_inputs, wi, wo) 
   wi, wo = back_propagation(h, wi, wo, train_inputs, train_outputs, y_pred, lr) 


y_pred, h = forward_propagation(train_inputs, wi, wo)
Y_train = train_outputs
error = error(Y_train, y_pred)
#rint(Y_train[0:20], y_pred[0:20])


X_test  = np.array([
  #([[0, 1, 0], [1, 1, 1], [0, 1, 0], [1, 1, 1], [0, 1, 0]]), 0),  # Ziffer 0 
   [[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]],  # Ziffer 1 
   [[0, 1, 1], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1]],  # Ziffer 2 
   [[1, 1, 1], [0, 0, 1], [0, 1, 1], [0, 0, 1], [1, 1, 1]],  # Ziffer 3 
   [[1, 0, 0], [1, 1, 0], [1, 1, 1], [0, 1, 0], [0, 1, 0]],  # Ziffer 4 
   [[1, 1, 1], [1, 0, 0], [1, 1, 1], [0, 0, 1], [1, 1, 1]],   # Ziffer 5 
])

yy, h = forward_propagation(X_test, wi, wo) 
Y_test= np.array([[1], [2], [3], [4], [5]])

Y_pred = [0] * len(Y_test)
    
for i in range(len(Y_test)):
   if (np.mean(yy) > 0.5): 
       Y_pred[i] = 1
   elif (np.mean(yy) <= 0.5):
       Y_pred[i] = 0
       
accuracy = accuracy_score(Y_test, Y_pred)

print(accuracy)



