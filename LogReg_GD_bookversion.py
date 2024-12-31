# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 15:58:38 2024

@author: srivi
"""
import numpy as np
import numpy.random as rand
import pandas as pd


class LogisticRegressionGD(object):

    def __init__(self, lr=0.05, n_iter=100, random_state=1):
        self.lr = lr
        self.n_iter = n_iter
        self.random_state = random_state
    
    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.lr * X.T.dot(errors)
            self.w_[0] += self.lr * errors.sum()
# Wir berechnen nun den Wert der Straffunktion
# der logistischen Regression, nicht mehr die
# Summe der quadrierten Werte der Straffunktion.
            cost = (-y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output))))
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """Netzeingabe berechnen"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, z):
        """Logistische Aktivierungsfunktion berechnen"""
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))
    def predict(self, X):
        """Klassenbezeichnung zurückgeben"""
        return np.where(self.net_input(X) >= 0.0, 1, 0)
# Entspricht:
# return np.where(self.activation(self.net_input(X))
# >= 0.5, 1, 0

X_train_01_subset = X_train[(y_train == 0) | (y_train == 1)]
y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]

lrgd = LogisticRegressionGD(eta=0.05, n_iter=1000, random_state=1)
lrgd.fit(X_train_01_subset, y_train_01_subset)
plot_decision_regions(X=X_train_01_subset, y=y_train_01_subset,classifier=lrgd)
plt.xlabel('Länge des Blütenblatts [standardisiert]')
plt.ylabel('Breite des Blütenblatts [standardisiert]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()


