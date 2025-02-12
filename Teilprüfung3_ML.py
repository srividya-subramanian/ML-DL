# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 10:45:15 2025

@author: srivi
"""
from matplotlib import pyplot as plt

#Generate a synthetic dataset with three classes and two characteristics per class
from sklearn.datasets import make_blobs
X, Y = make_blobs(n_samples=150, n_features=6, centers=3, cluster_std=[0.6, 0.9, 0.8], random_state=0)
plt.scatter(X[:,0], X[:,1])
plt.show()

#Split the data set into 70% training data and 30% test data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, train_size=0.7)

#decision tree classifier with maximum depth of the tree = 3
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion='gini', max_depth=3)
dt.fit(X_train, Y_train)
y_pred_dt = dt.predict(X_test)

# K-Nearest Neighbors with K=5 
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, Y_train)
y_pred_knn = knn.predict(X_test)

#accuracy of the predictions
from sklearn.metrics import accuracy_score
acc1 = accuracy_score(Y_test, y_pred_dt)
acc2 = accuracy_score(Y_test, y_pred_knn)

print(acc1, acc2)

print('Both Decision tree and K nearest neighbours has the best in this case with the accuracy of 1.0. ')
print('This could be because of the availability of good number of fearures that describes the classes')