# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 12:01:56 2025

@author: srivi
"""
import pandas as pd
import numpy as np


df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
df.columns = ['Klassenbezeichnung', 'Alkohol','Apfelsäure', 'Asche', 'Aschealkalität', 'Magnesium',
   'Phenole insgesamt', 'Flavanoide', 'Nicht flavanoide Phenole','Tannin',
   'Farbintensität', 'Farbe','OD280/OD315 des verdünnten Weins','Prolin']

print('Klassenbezeichnungen', np.unique(df['Klassenbezeichnung']))

df_wine = df

Y=df['Klassenbezeichnung']
X=df.drop(['Klassenbezeichnung'],axis=1)


'''Split the data set into 70% training data and 30% test data'''
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, train_size=0.7,stratify=Y)

