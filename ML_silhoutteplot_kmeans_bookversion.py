# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 11:30:41 2025

@author: srivi
"""

'''Generate datasets'''
from sklearn.datasets import make_blobs
X,y = make_blobs(n_samples=300, n_features=2, centers=3, cluster_std=2, random_state =42)

import matplotlib.pyplot as plt
plt.scatter(X[:,0], X[:,1], marker='o',c='lightpink',edgecolors='black',s=20)
plt.show()

from sklearn.cluster import KMeans 
km = KMeans(n_clusters= 3, init='k-means++', n_init='auto', max_iter=100, tol=0.0001, random_state=42)

y_km = km.fit_predict(X)
import numpy as np
from matplotlib import cm
from sklearn.metrics import silhouette_samples
cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(X,
    y_km,
    metric='euclidean')
y_ax_lower, y_ax_upper = 0, 0
yticks = []
for i, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[y_km == c]
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(float(i) / n_clusters)
    plt.barh(range(y_ax_lower, y_ax_upper),
    c_silhouette_vals,
    height=1.0,
    edgecolor='none',
    color=color)
    yticks.append((y_ax_lower + y_ax_upper) / 2.)
    y_ax_lower += len(c_silhouette_vals)
silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg,
    color="red",
    linestyle="--")
plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouettenkoeffizient')
plt.tight_layout()
plt.show()