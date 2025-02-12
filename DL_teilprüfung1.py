# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 08:53:07 2025

@author: srivi
"""

import numpy as np
import pandas as pd


'''Input (a list with two input values) '''
x = [0.5, -1.5]

'''Weights (a 3x2 matrix with weights)'''
w =np.array([[0.9,-0.1, 0.1],[-0.2, 0.8, 0.4]])

'''A Python function forward_propagation that takes two parameters: inputs 
 and weights (a 3x2 matrix with weights). The function should calculate
 the weighted sums for each output node and return them as a list. Extend your function 
by implementing the ReLU (Rectified Linear Unit) activation function and apply it to 
the output of your forward_propagation function before returning the result.'''

def forward_propagation(x, w):
    wsum = np.dot(x, w).tolist()
    relu =[]
    for i in wsum:
        mx = max(0,i)    
        relu.append(mx)
    return relu

y_pred = forward_propagation(x, w)
print(y_pred)
