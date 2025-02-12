# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 10:01:54 2025

@author: srivi
"""
import pandas as pd
import numpy as np
import re
'''
Null integer values = None, np.nan, pd.NA
'''

'''generate dataset'''
df = pd.DataFrame([
    ['1', None, 1,'bad'],
    ['2', 33, 3,'require improvement'],
    ['3', 54, None,'Super product'],
    ['4', 42, 4,'NAN'],
    ['5', None, 4,'good'],
    ['6', 45, 3,'!!!###'],
    ['7', 48, 5,'Excellent'],
    ['8', 44, None,'ok product'],
    ['9', 39, 1,'NAN'],
    ])

df.columns = ['ID','Age','Review','Comment']


'''Remove all lines where the `age` or `rating` is missing. '''
df1= df.dropna(subset=['Age','Review'])

''' Replace missing 'Comments' with the text: “No comment”'''
df2 = df1.replace({'NAN':'No comment'})

'''Remove all rows in which the 'Comment' consists only of special characters 
    (e.g. “!!!”, “???”, “###” etc.).  '''
df2['Comment'] = df2['Comment'].apply(lambda x: re.sub(r'^\W+', 'No Comment', x))


'''Add a new column 'Label' in which the ratings are classified as follows: 
    1-2 stars as 'Negative', 3 stars as 'Neutral' and 4-5 stars as 'Positive'. '''
df2['Label'] = df2['Review'].apply(lambda x: 'Negative' if x<=2 else 'Neutral' if x==3 else 'Positive')


def clean(df):
    df1= df.dropna(subset=['Age','Review'])   
    df2 = df1.replace({'NAN':'No comment'})
    df2['Comment'] = df2['Comment'].apply(lambda x: re.sub(r'^\W+', 'No Comment', x))
    df2['Label'] = df2['Review'].apply(lambda x: 'Negative' if x<=2 else 'Neutral' if x==3 else 'Positive')
    return df2

df_new = clean(df)    
    
df2 = df1.replace({'NAN':'No comment'})    
f = lambda x: re.sub(r'^\W+', 'No Comment', x)
f(df2['Comment'])