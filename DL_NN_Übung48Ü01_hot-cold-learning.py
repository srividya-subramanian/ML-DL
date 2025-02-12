# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 14:57:56 2025

@author: srivi
"""
import numpy as np
import pandas as pd

#Explain how hot and cold learning works using an example of your choice. You should describe the basic idea of hot and cold learning,  

''' Hot and cold learning is about adjusting weights - meaning increasing / decreasing weights so that the 
error is reduced to 0 with a very high accurate prediction. '''

'''define a simple neural network with a weight'''

def neural_network(x, w):
    wsum = np.dot(x, w)
    return wsum

def error(y, y_pred):
    err = (y_pred - y) ** 2
    return err

'''formulate an algorithm in pseudocode or Python code that adjusts this weight so that 
a given target value (goal_prediction) is predicted as accurately as possible by the network.  

Use the hot and cold learning method by gradually increasing or decreasing the weight and 
measuring the error. The error should be calculated as the squared deviation from the target value. 
Also state why it makes sense to only consider positive errors.  '''

x = [0.5]
y = [0.8]
w = 0.5
lr = 0.1

for iter in range(100):
    y_pred = neural_network(x, w)
    err = error(y, y_pred)
    print(err)
    
    upw = w+lr
    up_pred = neural_network(x, upw)
    up_err = error(y, up_pred)
    
    lw = w-lr
    lw_pred = neural_network(x, lw)
    lw_err = error(y, lw_pred)
    
    if up_err > lw_err: 
        w = w-lr 
        #print('weight decreased')
    else: 
        w = w+lr 
        #print('weight increased')
        
        
'''It makes sense to only consider positive errors because otherwise errors in opposite directions 
could cancel each other out. This would lead to a misleading average calculation in which a network 
could be falsely regarded as accurate, although there are actually large deviations from the target value. 
'''