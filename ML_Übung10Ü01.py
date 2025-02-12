# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 21:51:54 2025

@author: srivi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import heatmap, scatterplotmatrix
rng= np.random.default_rng(seed = 0)


np.random.seed(0)
column = ['CRIM','ZN','INDUS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO']
df = pd.DataFrame(rng.uniform(low=2.5, high=10, size=(100,10)), columns = list(column)).round(2)

cols=['CRIM','ZN','INDUS','NOX','RM','AGE']
variable = [1.2, 12.5, 10, 0.1, 1, 10, 1, 1, 100, 2]
df_new = df.mul(variable, axis=1)


scatterplotmatrix(df[cols].values, figsize=(20, 15), names=cols)
plt.show()

cm = np.corrcoef(df[cols].values.T)
hm = heatmap(cm, row_names=cols, column_names=cols)
plt.show()