# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 19:28:26 2024

@author: srividya subramanian
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.random as rand

array = rand.randint(0,50,100)
avg = np.mean(array)
min = min(array)
max = max(array)

array_new = []
for x in array:
    if x > avg: 
        array_new.append(x)
        print(x)
print(array_new)    

sorted_array = sorted(array_new)

