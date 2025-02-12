# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 17:25:28 2025

@author: srivi
"""

import pandas as pd
import numpy as np

df = pd.read_csv('https://raw.githubusercontent.com/'
    'rasbt/python-machine-learning-book-'
    '2nd-edition/master/code/ch10/housing.data.txt', sep='\s+')

df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS',
    'NOX', 'RM', 'AGE', 'DIS', 'RAD',
    'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

import matplotlib.pyplot as plt
from mlxtend.plotting import scatterplotmatrix
cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
scatterplotmatrix(df[cols].values, figsize=(10, 8),names=cols, alpha=0.5)
plt.tight_layout()
plt.show()


from mlxtend.plotting import heatmap
import numpy as np
cm = np.corrcoef(df[cols].values.T)
hm = heatmap(cm, row_names=cols, column_names=cols)
plt.show()


