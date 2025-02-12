# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 09:37:24 2025

@author: srivi
"""
'''Generate datasets'''
from sklearn.datasets import make_blobs
X,y = make_blobs(n_samples=300, n_features=2, centers=3, cluster_std=2, random_state =42)

import matplotlib.pyplot as plt
plt.scatter(X[:,0], X[:,1], marker='o',c='lightpink',edgecolors='black')
plt.show()

'''k-means cluster analysis for each k from 1 to 10 '''
from sklearn.cluster import KMeans 

'''a distortion measure - summarizes how the distances between the embedded
points deviate from the original distances'''
distortion = []
for i in range(1,10,1):
    km = KMeans(n_clusters= i, n_init='auto', max_iter=100, tol=0.0001, random_state=42)
    km.fit(X)
    distortion.append(km.inertia_)

'''Plot the SSE values for each k and use `matplotlib.pyplot` for visualization'''
    
plt.plot(range(1,10,1),distortion, marker='o')
plt.show()

print('Sum of squared deviation (SSE): %.2f' % km.inertia_)


