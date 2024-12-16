# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 11:39:56 2024

@author: srivi
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.random as rand
from math import e 

def activationfn(x):
    sig = 1 / (1+e**(-x))
    return sig

def derivativefn(x):
    dsig = e**(-x)/ (1+e**(-x))
    return dsig

x = pd.DataFrame([-5,-4, -3, -2, -1, 0, 1, 2, 3, 4, 5])

for x in x.values:
    sigmoid_fn = activationfn(x)
    derivative_sigmoid_fn = derivativefn(x)
    print(sigmoid_fn, derivative_sigmoid_fn)
