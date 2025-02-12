# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 15:38:26 2025

@author: srivi
"""

import numpy as np
import pandas as pd
from numpy.random import uniform as rand

toes = rand(8,10,4).round(1)
wlrec = rand(0,1,4).round(1)
nfans = rand(0.5,2,4).round(1)

weights = rand(0.1,0.7,3).round(1)
print(toes, wlrec, nfans, weights)
xx = np.column_stack((toes,wlrec,nfans))

x=xx[0]
def neural_network(x,w):
    pred = np.dot(x,w)
    return pred

x=xx[3]
print(neural_network(x,weights))
