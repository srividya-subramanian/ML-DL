# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 15:41:08 2025

@author: srivi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Generate a synthetic dataset 
from sklearn.datasets import make_blobs

X, Y = make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=0.4, random_state=42)

#Split the data set 
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, train_size=0.7)

#Implement the KNN algorithm
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)

#Train/test the model 
knn.fit(X_train, Y_train)

y_pred = knn.predict(X_test)

#evaluation metric
from sklearn.metrics import classification_report

report = classification_report(Y_test, y_pred)

from sklearn.metrics import accuracy_score

accuracy_score = accuracy_score(Y_test, y_pred)

#Visualize the decision boundary 

from mlxtend.plotting import plot_decision_regions
plot_decision_regions(X, Y, clf = knn)
plt.show()


