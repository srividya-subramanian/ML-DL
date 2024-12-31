# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 07:51:27 2024

@author: srivi
"""

import pandas as pd
import numpy as np

# Data import

df = pd.read_csv('/Users/srivi/Documents/Velptec_K4.0031_3.0_ML_Kursmaterial/data.csv')

# count the number of nulls in the data

missing_values = df.isnull()
df= df.fillna(0)
null_count = df.isnull().sum()
df_dropna = df.dropna()

#drop dulicates

df = df.drop_duplicates()

#average price of all properties in the dataset
avg_price = df['price'].mean()
print('Average price of all properties in the dataset: ',avg_price)

#index of the most expensive house (house_id) and the corresponding price

max_price = df['price'].max()
dfmax =df[df['price'] == df['price'].max()]
print('Index of the most expensive house (house_id) and the corresponding price: ',dfmax[['price']])

#Create a new column called price_per_sqft that calculates the price per square foot for each property (price / square foot). 

price_persqft = df['price']/df['area']
df['Price per sqft']=price_persqft.astype(int)

# Add a binary column called high_price that indicates whether the price of a property is above the average price (1 for yes, 0 for no). 
df['High price indicator']=df['price'].apply(lambda x: 1 if x > avg_price else 0)

df_new = df

# Data standardization of columns area and price_per_sqft.
def zscore(data):
    mean = data.mean()
    std = data.std()
    z_scores = (data - mean) / std
    return np.round(z_scores.astype(float),2)

zscore_price = zscore(df['price'])
zscore_area = zscore(df['area'])
df_new['price']  =  zscore_price
df_new['area'] = zscore_area

def minmax_scaling(data):
    min = data.min()
    max = data.max()
    print(min,max)
    data = (data - min) /(max - min)
    return data

bedrooms_mm = minmax_scaling(df['bedrooms'])
df_new['bedrooms'] = bedrooms_mm

def data_summary(dat):
    sum = pd.DataFrame([dat.min(), dat.max(), dat.isna().sum(), dat.nunique(), dat.dtypes, dat.mean()],index=['Min', 'Max', '#NA', '#Uniques', 'dtypes', 'Mean']) 
    return sum
                       
data_summary = data_summary(df_new)                   
                       
df_new.to_csv('/Users/srivi/Documents/Velptec_K4.0031_3.0_ML_Kursmaterial/clean_data.csv')                       
                       