# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 16:56:45 2025

@author: srivi
"""
import numpy as np
import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/'
   'rasbt/python-machine-learning-book-'
   '2nd-edition/master/code/ch10/'
   'housing.data.txt', header=None, sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS',
   'NOX', 'RM', 'AGE', 'DIS', 'RAD',
   'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df.head()
X = df['LSTAT'].values.reshape(-1, 1)
y = df['MEDV'].values


from sklearn.preprocessing import PolynomialFeatures
quadratic = PolynomialFeatures(degree=2)
cubic = PolynomialFeatures(degree=3)
X_quad = quadratic.fit_transform(X)
X_cubic = cubic.fit_transform(X)

from sklearn.linear_model import LinearRegression
regr = LinearRegression()

X_fit = np.arange(X.min(), X.max(), 1)[:, np.newaxis]

from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
regr = regr.fit(X, y)
y_lin_fit = regr.predict(X_fit)

linear_r2 = r2_score(y, regr.predict(X))

regr = regr.fit(X_quad, y)
y_quad_fit = regr.predict(quadratic.fit_transform(X_fit))
quadratic_r2 = r2_score(y, regr.predict(X_quad))

regr = regr.fit(X_cubic, y)
y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))
cubic_r2 = r2_score(y, regr.predict(X_cubic))
# Ergebnisse ausgeben
plt.scatter(X, y,
   label='Trainingdatenpunkte',
   color='lightgray')
plt.plot(X_fit, y_lin_fit,
   label='Linear (d=1), $R^2=%.2f$' % linear_r2,
   color='blue',
   lw=2,
   linestyle=':')
plt.plot(X_fit, y_quad_fit,
   label='Quadratisch (d=2), $R^2=%.2f$'
   % quadratic_r2,
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

X_log = np.log(X)
y_sqrt = np.sqrt(y)
# Merkmale anpassen
X_fit = np.arange(X_log.min()-1,
    X_log.max()+1, 1)[:, np.newaxis]
regr = regr.fit(X_log, y_sqrt)
y_lin_fit = regr.predict(X_fit)
linear_r2 = r2_score(y_sqrt, regr.predict(X_log))
# Ergebnisse ausgeben
plt.scatter(X_log, y_sqrt,
    label='Trainingsdatenpunkte',
    color='lightgray')
plt.plot(X_fit, y_lin_fit,
    label='Linear (d=1), $R^2=%.2f$' % linear_r2,
    color='blue',
    lw=2)

plt.xlabel('log(% der Bev. mit niedr. Sozialstatus [LSTAT])')
plt.ylabel('$\sqrt{Preis \; in \; \1000$ [MEDV]}$')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()