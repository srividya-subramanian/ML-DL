# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 10:48:53 2025

@author: srivi
"""
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(1)
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, -1)
plt.scatter(X_xor[y_xor == 1, 0],X_xor[y_xor == 1, 1],c='b', marker='x',label='1')



import matplotlib.pyplot as plt
import numpy as np
def gini(p):
    return (p)*(1 - (p)) + (1 - p)*(1 - (1-p))
def entropy(p):
    return - p*np.log2(p) - (1 - p)*np.log2((1 - p))
def error(p):
    return 1 - np.max([p, 1 - p])
x = np.arange(0.0, 1.0, 0.01)
ent = [entropy(p) if p != 0 else None for p in x]
sc_ent = [e*0.5 if e else None for e in ent]
err = [error(i) for i in x]
fig = plt.figure()
ax = plt.subplot(111)
for i, lab, ls, c, in zip([ent, sc_ent, gini(x), err],
        ['Entropie', 'Entropie (skaliert)','Gini-Koeffizient','Klassifikationsfehler'],
        ['-', '-', '--', '-.'],
        ['black', 'lightgray','red', 'green', 'cyan']):
   line = ax.plot(x, i, label=lab,
   linestyle=ls, lw=2, color=c)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=5, fancybox=True, shadow=False)
ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')
plt.ylim([0, 1.1])
plt.xlabel('p(i=1)')
plt.ylabel('Unreinheitsindex')
plt.show()

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
tree.fit(X_train, y_train)
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X_combined, y_combined, classifier=tree, test_idx=range(105,150))
plt.xlabel('Länge des Blütenblatts [cm]')
plt.ylabel('Breite des Blütenblatts [cm]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
dot_data = export_graphviz(tree_model,
    filled=True,
    rounded=True,
    class_names=['Setosa',
    'Versicolor'
    'Virginica'],
    feature_names=['petal length',
    'petal width'],
    out_file=None)
graph = graph_from_dot_data(dot_data)
graph.write_png('tree.png')

 from sklearn.ensemble import RandomForestClassifier
   forest = RandomForestClassifier(criterion='gini',
    n_estimators=25,
    random_state=1,
    n_jobs=2)
   forest.fit(X_train, y_train)
   plot_decision_regions(X_combined, y_combined,
    classifier=forest,
    test_idx=range(105,150))
   plt.xlabel('Länge des Blütenblatts [cm]')
   plt.ylabel('Breite des Blütenblatts [cm]')
   plt.legend(loc='upper left')
   plt.tight_layout()
   plt.show()