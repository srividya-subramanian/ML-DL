# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 13:34:00 2025

@author: srivi
"""

import numpy as np

w = np.array([
    [0.1, -0.2, 0.1],
    [0.05, 0.2, -0.1]
    ])
X = np.array([[0.5, -0.1, 0.2], 
              [0.1, 0.2, 0.3], 
              [-0.3, 0.4, 0.1]]) 
y = np.array([[0.4, 0.2], 
              [0.3, 0.5], 
              [0.5, 0.1]])
lr = 0.01

def forward_pass(x,w):
    w=w.transpose()
    pred = np.dot(x,w)
    return pred

def calculate_error(y, y_pred):
    err = (y - y_pred)**2
    return err

def update_weights(x, w, y, y_pred): 
    err = calculate_error(y, y_pred)
    w += (np.dot(x,err) * lr).transpose()
    return w

error = []
for i in range(100): 
    
    y_pred = forward_pass(X,w)
    err = calculate_error(y, y_pred)
    error.append(err)
    w = update_weights(X, w, y, y_pred)





