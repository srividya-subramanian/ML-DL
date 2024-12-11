# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 11:20:16 2024

@author: srivi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.random as rnd
import statsmodels.api as sm
import scipy.optimize as opt


def berechne_linreg_koeffizienten(number): 
  werbung = np.sort(rnd.randint(10,100,number))
  unitssold= np.sort(rnd.randint(200,500,number))
  a,b = np.polyfit(werbung,unitssold,1)
  return a, b, werbung, unitssold

a, b, werbung, unitssold = berechne_linreg_koeffizienten(15)

def visualize_data(a, b, werbung, unitssold):
    predicted = a*werbung+b
    plt.scatter(werbung, unitssold, color='blue', label='actual') 
    plt.plot(werbung, predicted , color='red', label='predicted') 
    plt.xlabel('werbung in Thousand Euro')
    plt.ylabel('Number of Units sold')
    plt.legend() 
    plt.show() 

visualize_data(a, b, werbung, unitssold)