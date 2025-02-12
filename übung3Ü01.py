# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 16:56:16 2024

@author: srivi
"""

import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt

#Read data
df = pd.read_csv("/Users/srivi/Documents/Velptec_K4.0031_3.0_ML_Kursmaterial/gender_classification.csv")
print(df.describe(exclude='number'))

#Data Summary
def data_summary(dat):
    sum = pd.DataFrame([dat.isna().sum(), dat.nunique()],index=['NAN', 'Uniques']) 
    return sum
                       
data_summary1 = data_summary(df)                   
data_summary2 = df.describe()
#print(data_summary1, data_summary2)


#label encoder
from sklearn import preprocessing 
label_encoder = preprocessing.LabelEncoder() 
standard_scaler=preprocessing.StandardScaler()

from sklearn.pipeline import Pipeline  
from sklearn.svm import SVC
svm= SVC(kernel='rbf', random_state=1,gamma=0.10, C=10.0)

df['gender']= label_encoder.fit_transform(df['gender']) 
Y = df['gender']
X = df.drop(['gender'],axis =1)

from sklearn import model_selection as model
X_train, X_test, Y_train,Y_test= model.train_test_split(X,Y, test_size=0.2, train_size=0.8)

Pipeline=Pipeline([
   ('scaler',standard_scaler),
#   ('label_encoder', label_encoder),
   ('classifier',svm)  
    ])

Pipeline.fit(X_train,Y_train)
y_pred = Pipeline.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(Y_test, y_pred))

from mlxtend.plotting import plot_decision_regions
plot_decision_regions(X_train.values,Y_train.values, clf=svm,filler_feature_values={2},filler_feature_ranges={2})