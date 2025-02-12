# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 17:09:02 2025

@author: srivi
"""

import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt

#Read data
df = pd.read_csv("/Users/srivi/Documents/Velptec_K4.0031_3.0_ML_Kursmaterial/gender_classification.csv")
print(df.describe(exclude='number'))

#label encoder
from sklearn import preprocessing 
label_encoder = preprocessing.LabelEncoder() 
standard_scaler=preprocessing.StandardScaler()

from sklearn.pipeline import Pipeline  
from sklearn.svm import SVC
svm= SVC(kernel='rbf', random_state=1,gamma=0.10, C=5)

df['gender']= label_encoder.fit_transform(df['gender']) 
Y = df['gender']
X = df.drop(['gender'],axis =1)

from sklearn import model_selection as model
X_train, X_test, Y_train,Y_test = model.train_test_split(X,Y, test_size=0.2, train_size=0.8)

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
tree.fit(X_train, Y_train)

y_pred = tree.predict(X_test)

X_combined = np.vstack((X_train, X_test))
Y_combined = np.hstack((Y_train, Y_test))

from sklearn.metrics import classification_report
print(classification_report(Y_test, y_pred))

