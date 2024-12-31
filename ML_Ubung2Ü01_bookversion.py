# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 04:41:17 2024

@author: srivi
"""

import numpy as np

def heaviside(x):
    return 1 if x >= 0 else 0

class Perceptron:
    def __init__(self, length, lr=0.01):
        self.weights=np.random.rand(length+1)*0.01
        self.lr = lr
        
    def predict(self,inputs):
        summation = np.dot(inputs,self.weights[1:])+self.weights[0]
        return heaviside(summation)
    
    def train(self, inputs, labels):
        for input, label in zip(inputs, labels):
            prediction = self.predict(input)
            self.weights[1:] += self.lr * (label-prediction) * input
            self.weights[0] += self.lr * (label - prediction)
            
     
inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
labels = np.array([[0],[0],[0],[1]])

P = Perceptron(length=2)   
P.train(inputs, labels)

for input, label in zip(inputs, labels):
    prediction = P.predict(input)
    print(label, prediction)