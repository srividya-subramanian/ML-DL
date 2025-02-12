# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 23:04:46 2025

@author: srivi
"""

from sklearn.datasets import make_blobs

X,y = make_blobs(n_samples=200, n_features=2, centers=4, cluster_std=0.60, random_state=42)


import matplotlib.pyplot as plt
plt.scatter(X[:,0], X[:,1], c='white',marker='o',edgecolor='black',s=10)
plt.grid()
#plt.tight_layout()
plt.show()


from sklearn.cluster import KMeans
distortions = []
for i in range(1, 11):
    km = KMeans(n_clusters=i,init='k-means++', n_init=10, max_iter=100,random_state=0)
    km.fit(X)
    distortions.append(km.inertia_)
plt.plot(range(1,11), distortions, marker='o')
plt.xlabel('Anzahl der Cluster')
plt.ylabel('Verzerrung')
plt.tight_layout()
plt.show()

print('Verzerrung: %.2f' % km.inertia_)
