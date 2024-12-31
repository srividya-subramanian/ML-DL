# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 13:28:39 2024

@author: srivi
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.random as rand


class Perceptron(object):
    
    def __init__(self, lr=0.01, n_iter=10):
        self.lr = lr              #learning rate
        self.n_iter = n_iter      #no of iterations

    def fit(self, X, y):
        self.w = np.zeros(1 + X.shape[1])       #bias+weights
        self.errors_ = []                       #errors

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.lr * (target - self.predict(xi))
                self.w[1:] += update * xi       #weights
                self.w[0] += update             #bias
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def predict(self, X):
        summation = np.dot(X, self.w[1:]) + self.w[0]
        return np.where(summation >= 0.0, 1, -1)



df = pd.read_csv(
'https://archive.ics.uci.edu/ml/''machine-learning-databases/iris/iris.data',header=None, encoding='utf-8')

df[4]=df[4].astype('category').cat.codes # Convert Categorical codes to Categorical values
# similarly: df['mycol'] = df['mycol'].astype('category')
#d = dict(enumerate(df['mycol'].cat.categories))

X = df.drop([4],axis=1).values
y = df[4].values

ppn = Perceptron(lr=0.1, n_iter=10)

ppn.fit(X, y)

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')

plt.tight_layout()
plt.show()


from matplotlib.colors import ListedColormap
def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))