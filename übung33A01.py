# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 20:33:24 2024

@author: srivi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.random as rnd
width = [8,3,7,3,8,7,8,2,7,4]
height = [8,8,7,9,6,8,7,7,7,8]
label = ["Apple", "Banana", "Apple", "Banana", "Banana", "Apple", "Apple", "Banana", "Apple", "Banana"]




def visualize_data(width, height, label):
    plt.scatter(width,height, color='bluwise', label='actual') 
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.legend() 
    plt.show() 
    
    
def classification(height, width, label):
    label_new=[]
    for i in range(len(height)): 
        ht = height[i]
        wd = width[i]
        lb = label[i]
        if ht > 1.2 * wd: 
            print('Banana', lb)
            label_new.append('Banana')
        else:
            print('Apple', lb)
            label_new.append('Apple')
    return label_new


visualize_data(width, height, label)
label_new = classification(height, width, label)

testwidth=[8,4,7,3,9]
testheight=[9,10,7,9,8]
testlabel = ["", "", "", "", ""]

testlabel = classification(testheight, testwidth, testlabel)



