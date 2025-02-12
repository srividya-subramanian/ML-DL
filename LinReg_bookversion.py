# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 09:01:44 2025

@author: srivi
"""

class LinearRegressionGD(object):
    def __init__(self, eta=0.001, n_iter=20):
        self.eta = eta
        self.n_iter = n_iter
    
    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
            return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    def predict(self, X):
        return self.net_input(X)


X = df[['RM']].values
y = df['MEDV'].values

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()
lr = LinearRegressionGD()
lr.fit(X_std, y_std)

plt.plot(range(1, lr.n_iter+1), lr.cost_)
plt.ylabel('Summe quadrierter Abweichungen')
plt.xlabel('Epochen')
plt.show()




def lin_regplot(X, y, model):
   plt.scatter(X, y, c='steelblue', edgecolor='white',
   s=70)
   plt.plot(X, model.predict(X), color='black', lw=2)
   return None

lin_regplot(X_std, y_std, lr)
plt.xlabel('Durchschnittliche Anzahl der Zimmer [RM] (standardisiert)')
plt.ylabel('Preis in 1000$ [MEDV] (standardisiert)')
plt.show()

num_rooms_std = sc_x.transform([5.0])
price_std = lr.predict(num_rooms_std)

print("Preis in 1000$s: %.3f" sc_y.inverse_transform(price_std)
print('Steigung: %.3f' % lr.w_[1])
print('Achsenabschnitt: %.3f' % lr.w_[0])

from sklearn.linear_model import LinearRegression
slr = LinearRegression()
slr.fit(X, y)
print('Steigung: %.3f' % slr.coef_[0])
print('Achsenabschnitt: %.3f' % slr.intercept_)

lin_regplot(X, y, slr)
plt.xlabel('Durchschnittliche Anzahl der Zimmer [RM]')
plt.ylabel('Preis in 1000$ [MEDV]')
plt.show()


# Hinzufügen eines Spaltenvektors mit Einsen
Xb = np.hstack((np.ones((X.shape[0], 1)), X))
w = np.zeros(X.shape[1])
z = np.linalg.inv(np.dot(Xb.T, Xb))
w = np.dot(z, np.dot(Xb.T, y))
print('Steigung: %.3f' % w[1])
print('Achsenabschnitt: %.3f' % w[0])

from sklearn.linear_model import LinearRegression
slr = LinearRegression()
slr.fit(X, y)
print('Steigung and Achsenabschnitt:' % slr.coef)

lin_regplot(X, y, slr)
plt.xlabel('Durchschnittliche Anzahl der Zimmer [RM]')
plt.ylabel('Preis in 1000$ [MEDV]')
plt.show()






