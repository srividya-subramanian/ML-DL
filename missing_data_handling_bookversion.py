# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 14:58:37 2025

@author: srivi
"""

import pandas as pd
df = pd.DataFrame([
    ['green', 'M', 10.1, 'class2'],
    ['red', 'L', 13.5, 'class1'],
    ['blue', 'XL', 15.3, 'class2']])
df.columns = ['color', 'size', 'price', 'classlabel']

size_mapping = {'XL': 3,'L': 2,'M': 1}
df['size'] = df['size'].map(size_mapping)


df = pd.DataFrame([
    ['green', 'M', 10.1, 'class2'],
    ['red', 'L', 13.5, 'class1'],
    ['blue', 'XL', 15.3, 'class2']])
df.columns = ['color', 'size', 'price', 'classlabel']

inv_size_mapping = {k: v for k, v in size_mapping.items()}
df['size'] = df['size'].map(inv_size_mapping)

import numpy as np
class_mapping = {label:idx for idx,label in enumerate(np.unique(df['classlabel']))}
df['classlabel'] = df['classlabel'].map(class_mapping)

inv_class_mapping = {v: k for k, v in class_mapping.items()}
df['classlabel'] = df['classlabel'].map(inv_class_mapping)

from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)
print(y)

X = df[['color', 'size', 'price']].values
color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])
                                 
from sklearn.preprocessing import OneHotEncoder
X = df[['color', 'size', 'price']].values
color_ohe = OneHotEncoder()
color_ohe.fit_transform(X[:, 0].reshape(-1, 1)).toarray()                                 
                                 
from sklearn.compose import ColumnTransformer
X = df[['color', 'size', 'price']].values
c_transf = ColumnTransformer([
    ('onehot', OneHotEncoder(), [0]),
    ('nothing', 'passthrough', [1, 2])
    ])
c_transf.fit_transform(X).astype(float)

df_dum= pd.get_dummies(df[['price', 'color', 'size']])

df = pd.DataFrame([['green', 'M', 10.1,'class2'],
    ['red', 'L', 13.5,'class1'],
    ['blue', 'XL', 15.3,'class2']])
df.columns = ['color', 'size', 'price', 'classlabel']

df['x > M'] = df['size'].apply(
    lambda x: 1 if x in {'L', 'XL'} else 0)
df['x > L'] = df['size'].apply(
    lambda x: 1 if x == 'XL' else 0)

del df['size']


      
                                 
