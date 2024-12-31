# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 13:46:11 2024

@author: srividya subramanian
"""

import numpy as np
import numpy.random as rand

def initialise():
    w = rand.uniform(0.001,0.01,size=(2,))#weights
    lr = 0.01 #learning rate
    b = 0 #bias
    x = np.array([[0,0],[0,1],[1,0],[1,1]])
    labels = np.array([[0],[0],[0],[1]])
    return x, labels, w, b, lr

def error(y, y_pred):
    err = y - y_pred
    return err

def perceptron (x,w,b):
    y = np.dot(x,w) + b
    return 1 if y >= 0 else 0 #heavyside function

def train(x, labels, w, b, lr):
    for x, y in zip(x, labels):
        y_pred = perceptron(x, w, b)
        err = y - y_pred
        w += lr * err * x
        b += lr * err
    return y, y_pred
    
x, l, w, b, lr = initialise()
y, y_pred = train(x, l, w, b, lr)

for x, y in zip(x, l):
    y_pred = perceptron(x, w, b)
    print(y_pred, y)
