# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 16:58:42 2025

@author: srivi
"""

import pandas as pd 
import numpy as np 

# Beispiel-DataFrame
# Beispiel-DataFrame 
data = {'A': [1, 2, np.nan, 4, 5], 
        'B': [5, np.nan, np.nan, 8, 10], 
        'C': [10, 20, 30, 40, 50]} 

df = pd.DataFrame(data) 

def find_null(df):
    print("Are there nulls ?:", df.isnull().any() ,'', sep=os.linesep)
    print("No of nulls in each column:", df.isnull().sum() ,'', sep=os.linesep)
    null_col = df.columns[df.isnull().any()] 
    print("Columns with null values:", null_col ,'', sep=os.linesep)
    med = df[null_col].median()
    df_new = df.fillna(value=med)
    return med, df_new


import os     
med, df_new = find_null(df)
print("Median of columns with null values:", med ,'', sep=os.linesep)
print("The modified DataFrame:", df_new ,'', sep=os.linesep)
