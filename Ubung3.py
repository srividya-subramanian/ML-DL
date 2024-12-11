# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 13:07:04 2024

@author: srivi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.random as rnd
import statsmodels.api as sm
import scipy.optimize as opt


Dollar = [50,75,20,100]
Euro = [45, 67.50, 18, 90]


Dollar=pd.DataFrame(Dollar).astype(float)
Euro=pd.DataFrame(Euro).astype(float)
df = pd.concat([Dollar,Euro],axis=1)
data = [(50.00, 45.00), (75.00, 67.50), (20.00, 18.00), (100.00, 90.00)] 


a = 1

def update_wechselkurs(Dollar, Euro, a): 
    predicted = Dollar * a
    error = Euro - predicted
    a_new = a + 0.5 * error / Dollar
    return a_new

erate = []
for dollar,euro in data :   
    a=update_wechselkurs(dollar, euro, a)
    predicted = dollar * a
    error = euro - predicted
    erate.append(a)

def visualize_data(Dollar,Euro):
    predicted = Dollar * a
    plt.scatter(Dollar, Euro, color='blue', label='actual') 
    plt.plot(Dollar, predicted , color='red', label='predicted') 
    plt.xlabel('Preis in Dollar') 
    plt.ylabel('Preis in Euro') 
    plt.legend() 
    plt.show() 


visualize_data(Dollar,Euro)
