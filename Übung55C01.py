# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 20:48:39 2024

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

def initialise(x):
    inputLayerSize, hiddenLayerSize, outputLayerSize = 2,1,1 
    wi = np.random.rand(inputLayerSize, hiddenLayerSize) 
    wo = np.random.rand(hiddenLayerSize, outputLayerSize) 
    return wi, wo

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
    errh = dy_pred * np.dot(wo.T)
    dh = errh * derivativefn(h)
    
    wi += x.T*np.dot(dh)*lr
    wo += h.T*np.dot(y_pred)*lr
    return wi, wo

def generate_data():
    
