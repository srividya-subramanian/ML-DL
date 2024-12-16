# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 17:52:27 2024

@author: srivi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.random as rand
from sklearn.linear_model import LinearRegression 

def perceptron (x1, x2, w1, w2, th): 
    y = x1*w1 + x2*w2
    if y > th:
        y=1
        return y
    else:
        y=0
        return y

th = np.round(rand.uniform(0.5,1,1),1)
w1 = np.round(rand.uniform(0.1,1,1),1)
w2 = np.round(rand.uniform(0.1,1,1),1)
th = float(th)
w1 = float(w1)
w2 = float(w2)   
    
d1 = pd.DataFrame([0.1,0.6,0.4,0.8,0.2,0.0,0.9,0.2,0.3,0.7])
d2 = pd.DataFrame([0.7,0.2,0.3,0.1,0.5,0.0,0.9,0.2,0.6,0.4])        
d3 = pd.DataFrame([1,1,0,1,1,0,1,0,1,1])

df= pd.concat([d1,d2,d3],axis=1, ignore_index=True)
df= df.sample(frac=1).reset_index(drop=True)
df.columns=["pixelbereich1","pixelbereich2","klassification"]

for input in df.values :
    x1,x2,label = input
    y = perceptron(x1, x2, w1, w2, th)
    error= label - y
    print(error)
    if error != 0:
        w1 += error * x1
        w2 += error * x2
    y = perceptron(x1, x2, w1, w2, th)
    error= label - y
