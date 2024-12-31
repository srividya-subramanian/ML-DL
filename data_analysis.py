# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 21:11:30 2024

@author: srividya subramanian
"""

import pandas as pd

columns = ['ID', 'Name', 'Age', 'City', 'Income']
values = {'ID': [1,2,3], 'Name':['x','y','z'], 'Age':[32,40,35], 'City':['München','München','München'], 'Income':[55000,50000, 65000]}

df = pd.DataFrame(values, columns=columns)

print(df.shape)
print(df.head(5))
print(df['Income'].mean())
print(df.columns)

  