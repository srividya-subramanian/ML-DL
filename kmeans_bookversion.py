# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 16:15:27 2025

@author: srivi
"""

from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=150,
    n_features=2,
    centers=3,
    cluster_std=0.5,
    shuffle=True,
    random_state=0)
import matplotlib.pyplot as plt
plt.scatter(X[:,0],
    X[:,1],
    c='white',
    marker='o',
    edgecolor='black',
    s=50)
plt.grid()
plt.tight_layout()
plt.show()


from sklearn.cluster import KMeans
km = KMeans(n_clusters=3,
    init='random',
    n_init=10,
    max_iter=100,
    tol=1e-04,
    random_state=0)
y_km = km.fit_predict(X)


plt.scatter(X[y_km==0,0],
    X[y_km==0,1],
    s=50, c='lightgreen',
    marker='s', edgecolor='black',
    label='Cluster 1')
plt.scatter(X[y_km==1,0],
    X[y_km==1,1],
    s=50, c='orange',
    marker='o', edgecolor='black',
    label='Cluster 2')
plt.scatter(X[y_km==2,0],
    X[y_km==2,1],
    s=50, c='lightblue',
    marker='v', edgecolor='black',
    label='Cluster 3')
plt.scatter(km.cluster_centers_[:,0],
    km.cluster_centers_[:,1],
    s=250, marker='*',
    c='red', edgecolor='black',
    label='Zentrum')
plt.legend(scatterpoints=1)
plt.grid()
plt.tight_layout()
plt.show()