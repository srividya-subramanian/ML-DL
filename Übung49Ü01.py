# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 21:53:23 2024

@author: srivi
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.random as rand
from sklearn.linear_model import LinearRegression 

def perceptron (x1, x2, w1, w2): 
    y = x1*w1 + x2*w2
    if y > 1:
        y=1
        return y
    else:
        y=0
        return y
    

w1 = np.round(rand.uniform(0.1,1,1),1)
w2 = np.round(rand.uniform(0.1,1,1),1)

testdata = [(0,0),(0,1),(1,0),(1,1)]
output = []

for x1, x2 in testdata:
    y = perceptron (x1, x2, w1, w2)
    output.append(y)
    print(x1,w1,x2,w2,y)

print(output)    


    