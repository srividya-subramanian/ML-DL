# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 12:44:08 2024

@author: srivi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.random as rand
from math import e 

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

inputLayerSize, hiddenLayerSize, outputLayerSize = 2, 3, 1 
wi = np.random.rand(inputLayerSize, hiddenLayerSize) 
wo = np.random.rand(hiddenLayerSize, outputLayerSize) 


train_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) 
train_outputs = np.array([[0], [1], [1], [0]]) 

lr = 0.1 
epochs = 10000 
for epoch in range(epochs): 
   y_pred, h = forward_propagation(train_inputs, wi, wo) 
   wi, wo = back_propagation(h, wi, wo, train_inputs, train_outputs, y_pred, lr) 

print(forward_propagation(train_inputs, wi, wo)[0])

