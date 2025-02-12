# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 15:33:24 2025

@author: srivi
"""

'''Generate a synthetic dataset '''
from sklearn.datasets import make_blobs
X,y = make_blobs(n_samples=300, n_features=2, centers=3, random_state=42)

from sklearn.cluster import AgglomerativeClustering
clustering = AgglomerativeClustering(n_clusters=3)
y_pred = clustering.fit_predict(X)

import matplotlib.pyplot as plt
colors = ['lightgreen', 'lightpink', 'lightblue']

plt.figure(figsize=(8,6))

for i in range(3):
    plt.scatter(X[y_pred == i,0], X[y_pred == i,1], s=50, c=colors[i],edgecolors='black')
#    plt.scatter(X[:,0], X[:,1], s=50, c=colors[i])
    cc= X[y_pred==i].mean(axis=0)
    print(i,cc)
    plt.scatter(cc[0],cc[1],s=300, marker='*', c='black')

plt.title('Agglomerative Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()


