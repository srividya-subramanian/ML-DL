# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 19:00:39 2024

@author: srivi
"""

import numpy as np
import pandas as pd
import numpy.random as rand

class LogisticRegressionGD(object):

    def __init__(self, lr=0.05, n_iter=100):
        self.lr = lr
        self.n_iter = n_iter
 
    def random_weights(self, X):
        return rand.RandomState(1).normal(loc=0.0, scale=0.01, size=X.shape[1])

    def random_bias(self, X):
        return rand.RandomState(1).normal(loc=0.0, scale=0.01, size=1)
    
    def net_input(self, X):
        return np.dot(X, self.w) + self.b

    def activation(self, z):
        return 1. / (1. + np.exp(-z))

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)

    
    def fit(self, X, y):
        self.w = self.random_weights(X)
        self.b = self.random_bias(X)
        self.cost_ = []
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w += self.lr * X.T.dot(errors)
            self.b += self.lr * errors.sum()
            cost = (-y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output))))
            self.cost_.append(cost)
        return self

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


def random_normal_distribution(mu,sig,n): # n is the length of the random array
    return rand.RandomState(1).normal(mu, sig, n)

arr1 =  random_normal_distribution(50,3,50)
l1 = np.zeros(50).astype(int)
df1 = pd.DataFrame(zip(arr1,l1))
df1.columns = ['data','label']
arr2 = random_normal_distribution(100,5,50) 
l2 = (l1+1).astype(int)
df2 = pd.DataFrame(zip(arr2,l2))
df2.columns = ['data','label']
df=pd.concat([df1,df2]).sample(frac=1).reset_index(drop=True)


from sklearn.datasets import make_classification

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

lrgd = LogisticRegressionGD()
y = lrgd.fit(X_train, y_train)

y_pred = lrgd.predict(X_test)
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1score = f1_score(y_test, y_pred)
                                                    
print(accuracy,precision,recall,f1score)


X_combined_std = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X=X_combined_std,y=y_combined,classifier=lrgd)
plt.xlabel('')
plt.ylabel('')
plt.legend(loc='upper left')
plt.show()






