# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 15:45:03 2024

@author: srivi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.random as rnd

#load data
data= pd.read_csv(CSV-Datei.csv)

#convert into dataframe and keep only the following structure
df = pd.DataFrame(data)
df_new = df[['x-Koordinate','y-Koordinate','Pflanzenart']]

class MyLeastSquares:
 
  def __init__(self):
 
    self.intercept = 0
    self.slope = 0
    
  def CalculateSlope(self, X, y):
    
    numerator = 0
    denominator = 0

    #Calculate Mean
    mean_X = np.mean(X)
    mean_y = np.mean(y)
 
    #Looping through each observation
    for i in range(len(X)):
      numerator += (X[i] - mean_X) * (y[i] - mean_y)
      denominator += (X[i] - mean_X)**2
 
    slope = numerator/denominator
    slope = round(slope[0], 3)
    return slope
  
  def CalculateIntercept(self, slope, X, y):

    #Caluclate Mean
    mean_X = np.mean(X)
    mean_y = np.mean(y)

    intercept = mean_y - slope * mean_X
    intercept = round(intercept, 3)
    return intercept
 
  def fit(self, X, y):
 
    self.slope = self.CalculateSlope(X, y)
    self.intercept = self.CalculateIntercept(self.slope, X, y)

  def predict(self, X):

    return self.slope * X + self.intercept