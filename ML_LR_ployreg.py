# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 15:11:41 2025

@author: srivi
"""
import numpy as np
X = np.array([258.0, 270.0, 294.0, 320.0, 342.0,
    368.0, 396.0, 446.0, 480.0, 586.0])[:, np.newaxis]
y = np.array([236.4, 234.4, 252.8, 298.6, 314.2,
    342.2, 360.8, 368.0, 391.2, 390.8])

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
lr = LinearRegression()

lr.fit(X,y)
y_pred = lr.predict(X)

X_fit = np.arange(250,600,10)[:, np.newaxis]
y_fit = lr.predict(X_fit)

qpoly = PolynomialFeatures(degree=2)
qX = qpoly.fit_transform(X)
lr.fit(qX, y)
qy_pred= lr.predict(qX)

import matplotlib.pyplot as plt
plt.xlim((200, 600))
plt.ylim((200, 500))
plt.scatter(X, y, label='Trainingsdatenpunkte')
plt.plot(X_fit, y_fit, label='Lineare Anpassung', linestyle='--')
plt.plot(qX, qy_pred, label='Quadratische Anpassung')
plt.xlabel('Erklärende Variable')
plt.ylabel('Vorhergesagte/bekannte Zielwerte')
plt.show()

from sklearn.metrics import r2_score

print('R^2-Training linear: %.3f, quadratisch: %.3f'
   % ( r2_score(y, y_pred), r2_score(y, qy_pred)))