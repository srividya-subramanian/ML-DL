# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 14:56:24 2025

@author: srivi
"""

import pandas as pd
df = pd.DataFrame([
    ['green', 'M', 10.1],
    ['red', 'L', 13.5],
    ['blue', 'XL', 15.3],
    ['red', 'L', 13.8],
    ['blue', 'XL', 15.7],
    ['green', 'M', 10.9],
    ['red', 'L', 13.1],
    ['blue', 'XL', 15.0],
    ['green', 'M', 10.5],
       ])
df.columns = ['color', 'size', 'price']
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer

X = df[['color', 'size', 'price']].values
c_transf = ColumnTransformer([
    ('onehot', OneHotEncoder(), [0]),
    ('nothing', 'passthrough', [1, 2])
    ])
print(c_transf.fit_transform(X))


ohe = OneHotEncoder()
df_ohe = ohe.fit_transform(df)
#print(df.toarray())

ct = ColumnTransformer(transformers = [('ohe_col1', OneHotEncoder(),[0]),#['color']
                                     ('ord_col2', OrdinalEncoder(),[1]),#['size']
                                    ('nothing','passthrough',['price']) ])
df_ct = ct.fit_transform(df)
print(df_ct)


ct1 = ColumnTransformer([('ohe_col1',OneHotEncoder(drop='first'),['color']),('nothing','passthrough',['size', 'price']) ])
df_ct1 = ct1.fit_transform(df)
print(df_ct1)
