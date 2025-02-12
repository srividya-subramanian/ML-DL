# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 14:54:03 2025

@author: srivi
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler  


col = ['Zimmer', 'Preis']

df = pd.DataFrame({
    'Zimmer':[5,4,6,3,7], 
    'Preis':[300,250,350,200,400]
    })
    
scaler = StandardScaler()
scaled = scaler.fit_transform(df.to_numpy())
df_scaled = pd.DataFrame(scaled, columns=col)
X_std = df_scaled['Zimmer'].values
y_std = df_scaled['Preis'].values

class GD(object):
    def __init__(self, lr=0.0001, n_iter=20):
        self.lr = lr
        self.n_iter = n_iter
        self.b = 0
        self.w = 0

    def error(self, y,y_pred):
        err = y - y_pred
        return err
    
    def perceptron(self, X):
        y = np.dot(X,self.w) + self.b
        print(y)
        return y
    
    def fit(self, X, y):
        self.cost = []
#        self.cost = np.zeros(n_iter)

        for i in range(self.n_iter):
            y_pred = self.perceptron(X)
            error  = self.error(y,y_pred)
            #print(i, error)
            self.w += np.dot(X.T, error)*self.lr
            self.b += error.sum()*self.lr
            c = (error**2).sum() / 2.0
            self.cost.append(c)
#            self.cost[i] = c
        return self
        
LR = GD()
LR.fit(X_std, y_std)

plt.plot(range(1,LR.n_iter+1), LR.cost)
plt.show()
    
    
    