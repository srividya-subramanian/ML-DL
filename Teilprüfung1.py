# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 08:32:36 2024

@author: srivi
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.random as rand
from sklearn.linear_model import LinearRegression 


def generate_random_data(number): 
  label1=[]
  label2=[]

  RosenBlütenblattgröße = pd.DataFrame(np.round(rand.uniform(2,8,number),1))
  TulpenBlütenblattgröße = pd.DataFrame(np.round(rand.uniform(5,11,number),1))
  RosenFarbintensität  = pd.DataFrame(rand.randint(10,255,number))
  TulpenFarbintensität =  pd.DataFrame(rand.randint(10,255,number))
  for i in range(10): label1.append('1') #Rose
  label1=pd.DataFrame(label1)
  for i in range(10): label2.append('2') #Tulip
  label2=pd.DataFrame(label2)
  rose = pd.concat([RosenFarbintensität, RosenBlütenblattgröße, label1], axis=1)
  tulip = pd.concat([TulpenFarbintensität,TulpenBlütenblattgröße,label2], axis=1)
  df= pd.concat([rose,tulip],axis=0, ignore_index=True)
  df= df.sample(frac=1).reset_index(drop=True)
  df.columns=["Farbintensität","Blütenblattgröße","Klasse"]
  return df


df=generate_random_data(15)
train = df.dropna().reset_index(drop=True)
test = df.loc[df['Klasse'].isna()].reset_index(drop=True)

def visualize_data(df):
    xx=df['Farbintensität']
    yy=df['Blütenblattgröße']
    ll=df['Klasse'].to_numpy()
    plt.scatter(xx, yy, c=ll.astype(float))
    plt.xlabel('Farbintensität')
    plt.ylabel('Blütenblattgröße')
   # plt.legend() 
    plt.show() 

visualize_data(train)

X = train[['Farbintensität','Blütenblattgröße']]
y = train[['Klasse']]
model = LinearRegression()
model.fit(X, y)

testX=test.drop(['Klasse'], axis=1)
testY = pd.DataFrame(model.predict(testX))

prediction =  pd.concat([testX, round(testY).astype(int)],axis=1, ignore_index=True)
prediction.columns=["Farbintensität","Blütenblattgröße","Klasse"]

visualize_data(prediction)



