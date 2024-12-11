# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 13:36:16 2024

@author: srivi
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier


df = pd.read_csv('gender_classification.csv')
df.head(5)

columns = df.columns

X = df.drop('gender', axis=1)
Y = df['gender']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

print(X.shape, Y.shape, X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

classifier = DecisionTreeClassifier()

classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
precision = precision_score(Y_test, Y_pred, pos_label='Female')
recall = recall_score(Y_test, Y_pred, pos_label='Female')
f1score = f1_score(Y_test, Y_pred, pos_label='Female')

print(accuracy, precision, recall, f1score)


knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)

accuracy = accuracy_score(Y_test, Y_pred)
precision = precision_score(Y_test, Y_pred, pos_label='Female')
recall = recall_score(Y_test, Y_pred, pos_label='Female')
f1score = f1_score(Y_test, Y_pred, pos_label='Female')

print(accuracy, precision, recall, f1score)


