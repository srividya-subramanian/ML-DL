# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 14:23:29 2024

@author: srivi
"""


import numpy as np
import pandas as pd
import numpy.random as rand
from matplotlib.colors import ListedColormap 
import matplotlib.pyplot as plt

def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],alpha=0.8, c=colors[idx],marker=markers[idx], label=cl)


from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

X, y = make_classification(
    n_samples=100, # 1000 observations 
    n_features=2, # 5 total features
    n_informative=2, # 3 'useful' features
    n_redundant=0,
    random_state=1 # if you want the same results as mine
) 

from sklearn import model_selection as model

X_train, X_test, y_train, y_test = model.train_test_split(X, y, test_size=0.2)

from sklearn.preprocessing import StandardScaler 

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test  = ss.fit_transform(X_test)

lr = LogisticRegression(solver = 'lbfgs', C=1, random_state=1)
y = lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1score = f1_score(y_test, y_pred)
                                                    
print(accuracy,precision,recall,f1score)

X_combined_std = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X=X_combined_std,y=y_combined,classifier=lr)
plt.xlabel('')
plt.ylabel('')
plt.legend(loc='upper left')
plt.show()
