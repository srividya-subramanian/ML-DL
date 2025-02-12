# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 12:02:09 2025

@author: srivi
"""
import numpy as np
import pandas as pd


def neural_network(input, weights):
    out = 0
    for i in range(len(input)):
        out += (input[i] * weights[i])
    return out
def ele_mul(scalar, vector):
    out = [0,0,0]
    for i in range(len(out)):
        out[i] = vector[i] * scalar
    return out

toes = [8.5, 9.5, 9.9, 9.0]
wlrec = [0.65, 0.8, 0.8, 0.9]
nfans = [1.2, 1.3, 0.5, 1.0]
win_or_lose_binary = [1, 1, 0, 1]
true = win_or_lose_binary[0]
alpha = 0.3
weights = [0.1, 0.2, -.1]
input = [toes[0],wlrec[0],nfans[0]]

for iter in range(3):
    pred = neural_network(input,weights)
    error = (pred - true) ** 2
    delta = pred - true
    weight_deltas=ele_mul(delta,input)
    weight_deltas[0] = 0
    print("Iteration:" + str(iter+1))
    print("Prediction:" + str(pred))
    
    
    
    
def gradw(x, y, y_pred):
    sum = 0
    n = len(x)
    for i in range (0,n):
        err = (y_pred - y) ** 2
        sum += -x[i] * err
    dw =   sum 
    #print('delw', dw)
    return dw

pred = neural_network(input,weights)
error = (pred - true) ** 2
delta = pred - true
   
gradw(input, true, pred)


    