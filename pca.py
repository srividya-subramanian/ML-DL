# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 08:51:00 2025

@author: srivi
"""

import pandas as pd
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'   'machine-learning-databases/wine/wine.data',   header=None)
#

#df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',header=None)
#f = pd.read_csv('Ihr/lokaler/Pfad/zu/wine.data',header=None)


from sklearn.model_selection import train_test_split
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3,
    stratify=y,
    random_state=0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)


import numpy as np
cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print('\nEigenwerte \n%s' % eigen_vals)


tot = sum(eigen_vals)
var_exp = [(i / tot) for i in
    sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
import matplotlib.pyplot as plt
plt.bar(range(1,14), var_exp, alpha=0.5, align='center',
    label='Individuelle erklärte Varianz')
plt.step(range(1,14), cum_var_exp, where='mid',
    label='Kumulative erklärte Varianz')
plt.ylabel('Anteil an der erklärten Varianz')
plt.xlabel('Hauptkomponenten')
plt.legend(loc='best')
plt.tight_layout()
plt.show()


# Liste von (eigenvalues, eigenvector)-Tupel anlegen
eigen_pairs =[(np.abs(eigen_vals[i]),eigen_vecs[:,i])
    for i in range(len(eigen_vals))]
# Die Tupel in absteigender Reihenfolge sortieren
eigen_pairs.sort(key=lambda k: k[0], reverse=True)

w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
    eigen_pairs[1][1][:, np.newaxis]))
print('Matrix W:\n', w)


X_train_pca = X_train_std.dot(w)

colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_pca[y_train==l, 0],
    X_train_pca[y_train==l, 1],
    c=c, label=l, marker=m)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()


from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
# Initialisierung des PCA-Transformers und des
# Schätzers der logistischen Regression
pca = PCA(n_components=2)
lr = LogisticRegression(multi_class='ovr',
    random_state=1,
    solver='lbfgs')
# Dimensionsreduktion:
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
# Anpassung des logistischen Regressionsmodells
# an die verkleinerte Datenmenge:
lr.fit(X_train_pca, y_train)
plot_decision_regions(X_train_pca, y_train,
classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()
plot_decision_regions(X_test_pca, y_test, classifier=lr)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()


pca = PCA(n_components=None)
X_train_pca = pca.fit_transform(X_train_std)
pca.explained_variance_ratio_


