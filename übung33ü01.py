# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 08:48:31 2024

@author: srivi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.random as rnd

# Generate random, but relevant data

def generate_random_data(number): 
  label1=[]
  label2=[]

  heighthund = pd.DataFrame(rnd.randint(20,70,number))
  heightmensch = pd.DataFrame(rnd.randint(50,200,number))
  widthhund = pd.DataFrame(rnd.randint(40,80,number))
  widthmensch=  pd.DataFrame(rnd.randint(40,100,number))
  for i in range(10): label1.append('Hund')
  label1=pd.DataFrame(label1)
  for i in range(10): label2.append('Mensch')
  label2=pd.DataFrame(label2)
  hund = pd.concat([widthhund,heighthund,label1], axis=1)
  mensch = pd.concat([widthmensch,heightmensch,label2], axis=1)
  df= pd.concat([hund,mensch],axis=0, ignore_index=True)
  df= df.sample(frac=1).reset_index(drop=True)
  df.columns=["width","height","label"]
  return df, hund, mensch

# Data visualization

def visualize_data(df):
#    predicted = a*werbung+b
    plt.scatter(df['width'], df['height'], color='blue', label='actual') 
#    plt.plot(werbung, predicted , color='red', label='predicted') 
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.legend() 
    plt.show() 

# linear classification

def classification(a, height, width, label):
    label_new=[]
    for i in range(len(height)): 
        ht = height[i]
        wd = width[i]
        lb = label[i]
        if ht < a* wd: 
            print(i,ht,a* wd,'Hund',lb)
            label_new.append('Hund')
        else:
            print(i,ht,a* wd,'Mensch',lb)
            label_new.append('Mensch')
    return label_new

df, hund, mensch = generate_random_data(10)

height = df['height'].to_list()
width = df['width'].to_list()
label = df['label'].to_list()

visualize_data(df)
a = float(input('start valiue of gradient of dividing line: '))
classification(a, height,width, label)

testdata = [(78, 50), (60, 120), (82, 55), (55, 130), (88, 45)] 
height=[]
width=[]
label=[]
for i in testdata:
    i=list(i)
    width.append(i[0])
    height.append(i[1])
    label.append('')
    
final_label= classification(a, height, width, label)

        










