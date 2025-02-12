# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 14:59:02 2025

@author: srivi
"""

import pandas as pd
import numpy as np
np.random.seed(123)
variables = ['X', 'Y', 'Z']
labels = ['ID_0','ID_1','ID_2','ID_3','ID_4']
X = np.random.random_sample([5,3])*10
df = pd.DataFrame(X, columns=variables, index=labels)

from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

row_dist = pd.DataFrame(squareform(
    pdist(df, metric='euclidean')),
    columns=labels, index=labels)
row_dist

from scipy.cluster.hierarchy import linkage
row_clusters = linkage(row_dist,
    method='complete',
    metric='euclidean')
row_clusters = linkage(pdist(df, metric='euclidean'),
    method='complete')

row_clusters = linkage(df.values,
    method='complete',
    metric='euclidean')

pd.DataFrame(row_clusters,
    columns=['Zeile 1',
    'Zeile 2',
    'Distanz',
    '# Objekte im Cluster'],
    index=['Cluster %d' %(i+1) for i in
    range(row_clusters.shape[0])])

from scipy.cluster.hierarchy import dendrogram
# Dendrogrammfarbe Schwarz (Teil 1/2)
# from scipy.cluster.hierarchy import set_link_color_palette
# set_link_color_palette(['black'])
row_dendr = dendrogram(row_clusters,
    labels=labels,
    # Dendrogrammfarbe Schwarz (Teil 2/2)
    # color_threshold=np.inf
    )
plt.tight_layout()
plt.ylabel('Euklidische Distanz')
plt.show()

fig = plt.figure(figsize=(8,8), facecolor='white')
axd = fig.add_axes([0.09,0.1,0.2,0.6])
row_dendr = dendrogram(row_clusters, orientation='left')
# Für matplotlib < v1.5.1 orientation='right' verwenden
df_rowclust = df.iloc[row_dendr['leaves'][::-1]]
axm = fig.add_axes([0.23,0.1,0.6,0.6])
cax = axm.matshow(df_rowclust, interpolation='nearest', cmap='hot_r')
axd.set_xticks([])
axd.set_yticks([])
for i in axd.spines.values():
    i.set_visible(False)
fig.colorbar(cax)
axm.set_xticklabels([''] + list(df_rowclust.columns))
axm.set_yticklabels([''] + list(df_rowclust.index))
plt.show()



# Generate the linkage matrix
Z = linkage(X, method='complete')

# Plot the dendrogram
plt.figure(figsize=(10, 7))
dendrogram(Z, orientation='left')
plt.title('Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()


