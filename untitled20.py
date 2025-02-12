# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 14:30:41 2025

@author: srivi
"""

distortion = []
for i in range(1,10):
    km = KMeans(n_clusters=i,init="k-means++", n_init="auto", max_iter=300, random_state=42)
    y_km = km.fit(X)
    distortion.append(y_km.inertia_)
    
import matplotlib.pyplot as plt
plt.scatter(X[:,0], X[:,1])
plt.show()


plt.plot(range(1,10),distortion, marker='o',color = 'darkorange')
plt.xlabel('No. of clusters')
plt.ylabel('Distortion (SSE)')
plt.title('Elbow Diagram')
plt.show()
    
    