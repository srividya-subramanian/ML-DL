# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 11:29:37 2025

@author: srivi
"""

import numpy as np
import pandas as pd

from numpy.random import uniform
from numpy.random import normal

''' Generate data set '''
X = uniform(low=0,high = 10,size=100).round(1)
noise = normal(loc=0,scale=1,size=100).round(1)
y= X+noise

''' a scatter plot to visualize the generated data '''
import matplotlib.pyplot as plt
plt.scatter(X, y)
plt.ylabel('y=X+noise')
plt.xlabel('X')
plt.show()

''' Split the data set into a training and a test data set '''
XX=X.reshape(-1,1)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(XX,y,test_size=0.2)

'''Train a linear regression model '''
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
print(lr.score(X_test, y_test)) 

y_pred = lr.predict(X_test)

''' Evaluate the model with the R2 score'''
from sklearn.metrics import r2_score
r2 = r2_score(X_test, y_pred)
print(r2)

'''regression line together with the test data'''

plt.scatter(X_test, y_test)
plt.ylabel('y')
plt.xlabel('X')
plt.plot(X_test, y_pred, color ='k') 
plt.show()

