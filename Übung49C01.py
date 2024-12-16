# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 14:18:45 2024

@author: srivi
"""

import pandas as pd
import numpy as np


def activationfn (x1, x2): 
    if x1 > 0.5 or x2 > 0.5:
        a = 1
    else :
        a = 0
    return a

def perceptron (x1, x2): 
    w1 = 1
    w2 = 1
    th = 0.5
    y = x1*w1 + x2*w2
    if y > th: 
        y = 1 
    else:
        y = 0
    return y
    
    
d1 = pd.DataFrame([0.1,0.6,0.4,0.8,0.2,0.0,0.9,0.2,0.3,0.7])
d2 = pd.DataFrame([0.7,0.2,0.3,0.1,0.5,0.0,0.9,0.2,0.6,0.4])        
d3 = pd.DataFrame([1,1,0,1,1,0,1,0,1,1])

df= pd.concat([d1,d2,d3],axis=1, ignore_index=True)
df= df.sample(frac=1).reset_index(drop=True)
df.columns=["pixelbereich1","pixelbereich2","klassification"]

for input in df.values :
    x1,x2,label = input
    y = perceptron(x1, x2)
    print(y, label)

