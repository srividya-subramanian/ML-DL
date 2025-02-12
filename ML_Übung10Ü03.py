# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 15:11:41 2025

@author: srivi
"""
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
from sklearn.preprocessing import PolynomialFeatures
quad = PolynomialFeatures(degree=2)
cubic = PolynomialFeatures(degree=3)

import numpy.random as rand
rand.seed(42)
XX = rand.uniform(-3, 3, 30).round(2)
noise = rand.normal(0,1,30)
y = (1+(2*XX)+(XX**2)+noise).round(2)
X = XX.reshape(-1, 1)
X_fit = np.linspace(-3,3,200).reshape(-1, 1)
#np.arange(X.min(), X.max(), 1)[:, np.newaxis]


lr=lr.fit(X,y)
y_pred = lr.predict(X)
linear_r2 = r2_score(y, y_pred)

y_lin_fit = lr.predict(X_fit)

quadratic = PolynomialFeatures(degree=2)
X_quad = quadratic.fit_transform(X)
lrfit = lr.fit(X_quad, y)
qX_fit = quadratic.fit_transform(X_fit)
y_quad_fit = lrfit.predict(qX_fit)
quad_r2 = r2_score(y, lrfit.predict(X_quad))

cubic = PolynomialFeatures(degree=3)
X_cubic = cubic.fit_transform(X)
lrfit = lr.fit(X_cubic, y)
cX_fit = cubic.fit_transform(X_fit)
y_cubic_fit = lrfit.predict(cX_fit)
cubic_r2 = r2_score(y, lrfit.predict(X_cubic))



import matplotlib.pyplot as plt
 
plt.scatter(X, y,
   label='Trainingdatenpunkte',
   color='lightgray')
#plt.plot(X_fit, y_lin_fit,   label='Linear (d=1), $R^2=%.2f$' % linear_r2,  color='blue',  lw=2,  linestyle=':')
plt.plot(X_fit, y_quad_fit,
   label='Quadratisch (d=2), $R^2=%.2f$'
   % quad_r2,
   color='red',
   lw=2,
   linestyle='-')
plt.plot(X_fit, y_cubic_fit,
   label='Kubisch (d=3), $R^2=%.2f$' % cubic_r2,
   color='green',
   lw=2,
   linestyle='--')
plt.xlabel('% der Bev. mit niedr. Sozialstatus [LSTAT]')
plt.ylabel('Preis in 1000$ [MEDV]')
plt.legend(loc='upper right')
plt.show()

 