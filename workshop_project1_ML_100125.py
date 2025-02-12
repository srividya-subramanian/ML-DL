# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 09:23:05 2025

@author: srivi
"""
import numpy as np
import pandas as pd
#Read data
df = pd.read_csv("/Users/srivi/Downloads/wine+quality/winequality-red.csv", delimiter=';')

pd.set_option('display.max_columns',None)

#print(df.describe)#(exclude='number'))
#print(df.head(5))

import matplotlib.pyplot as plt
cor = df.corr()

plt.imshow(cor, cmap='Blues')
plt.colorbar()

# Adding labels to the cor
variables = []
for i in cor.columns:
    variables.append(i)

plt.xticks(range(len(cor)), variables, rotation=90)
plt.yticks(range(len(cor)), variables)
plt.show()

quality = abs(cor["quality"])>0.3


Y=df['quality']
X=df.drop(['quality'],axis=1)

#Split the data set into 70% training data and 30% test data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, train_size=0.7,stratify=Y)

#decision tree classifier with maximum depth of the tree = 3
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion='gini', max_depth=5,random_state= 0)
dt.fit(X_train, Y_train)
y_pred_dt = dt.predict(X_test)

# K-Nearest Neighbors with K=5 
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, Y_train)
y_pred_knn = knn.predict(X_test)

from sklearn.svm import SVC
svm= SVC(kernel='rbf', random_state=1,gamma=0.1, C=500)
svm.fit(X_train,Y_train)
y_pred_svm = svm.predict(X_test)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, criterion='gini',random_state= 0)
rf.fit(X_train,Y_train)
y_pred_rf = rf.predict(X_test)

#accuracy of the predictions
from sklearn.metrics import accuracy_score
acc1 = accuracy_score(Y_test, y_pred_dt)
acc4 = accuracy_score(Y_test, y_pred_knn)
acc3 = accuracy_score(Y_test, y_pred_svm)
acc2 = accuracy_score(Y_test, y_pred_rf)

print('Accuracy DT:',acc1,'     Accuracy RF:',acc2,'    Accuracy SVM:',acc3,'       Accuracy KNN:',acc4)

#confusion matrix
#acc = [acc1,acc2,acc3,acc4]
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred_rf)
print(cm)

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
feat_labels = df.columns[0:]
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))
plt.title('Bedeutung der Merkmale')
plt.bar(range(X_train.shape[1]), importances[indices], align='center')
plt.xticks(range(X_train.shape[1]), feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

models = []
#models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LR', LogisticRegression(solver='liblinear')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier(n_neighbors=6)))
models.append(('DT', DecisionTreeClassifier(criterion='gini', max_depth=5,random_state= 0)))
models.append(('RF', RandomForestClassifier(n_estimators=1000, criterion='gini',random_state= 0)))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto',kernel='rbf', random_state=1, C=500)))


# evaluate each model in turn
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=4, random_state=0, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
# Compare Algorithms
plt.boxplot(results, tick_labels=names)
plt.title('Algorithm Comparison')
plt.show()




from sklearn.model_selection import GridSearchCV, KFold
# Number of random trials
NUM_TRIALS = 8
 
# Set up possible values of parameters to optimize over
'''Best to choose max_features="sqrt" (using a random subset of size sqrt(n_features)) for classification tasks
(where n_features is the number of features in the data). The default value of max_features=1.0 is equivalent
to bagged trees and more randomness can be achieved by setting smaller values
(e.g. 0.3 is a typical default in the literature). Good results are often achieved when setting max_depth=None
in combination with min_samples_split=2 (i.e., when fully developing the trees).'''
rfc = RandomForestClassifier(max_depth=None)
p_grid = {"n_estimators": [100, 500], "max_features": [0.3, 1, 3], "min_samples_split": [2, 5]}
 
 
# Arrays to store scores
non_nested_scores = np.zeros(NUM_TRIALS)
 
# Loop for each trial
for i in range(NUM_TRIALS):
    # Choose cross-validation techniques for the inner and outer loops,
    # independently of the dataset.
    # E.g "GroupKFold", "LeaveOneOut", "LeaveOneGroupOut", etc.
    #inner_cv = KFold(n_splits=4, shuffle=True, random_state=i)
    outer_cv = KFold(n_splits=4, shuffle=True, random_state=i)
 
    # Non_nested parameter search and scoring
    clf = GridSearchCV(estimator=rfc, param_grid=p_grid, cv=outer_cv)
    clf.fit(X_train, Y_train)
    non_nested_scores[i] = clf.best_score_
 
# Plot scores on each trial for non-nested CV
plt.figure()
(non_nested_scores_line,) = plt.plot(non_nested_scores, color="r")
 
plt.ylabel("score", fontsize="14")
plt.title(
    "Non-Nested Cross Validation on Wine Quality Dataset",
    x=0.5,
    y=1.1,
    fontsize="15",
)
plt.show()



'''from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import Metadata
 
metadata = Metadata.detect_from_dataframe(
    data=df,
    table_name='wine_quality')
real_data = df
synthesizer = GaussianCopulaSynthesizer(metadata)
synthesizer.fit(real_data)
 
synthetic_data = synthesizer.sample(num_rows=100)'''


from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import Metadata
 
metadata = Metadata.detect_from_dataframe(
    data=df,
    table_name='wine_quality')
real_data = df
synthesizer = GaussianCopulaSynthesizer(metadata)
synthesizer.fit(real_data)
 
synthetic_data = synthesizer.sample(num_rows=100)

custom_synthesizer = GaussianCopulaSynthesizer(
    metadata,
    default_distribution='truncnorm',
    numerical_distributions={}
)
 
custom_synthesizer.fit(real_data)
 
from sdv.sampling import Condition
 
high_quality = Condition(
    num_rows=5,
    column_values={'quality': 8}
)
 
low_quality = Condition(
    num_rows=5,
    column_values={'quality': 3}
)

simulated_synthetic_data = custom_synthesizer.sample_from_conditions(conditions=[
  high_quality,
  low_quality
])

from sdmetrics.visualization import get_column_plot 
fig = get_column_plot(
    real_data=real_data,
    synthetic_data=simulated_synthetic_data,
    column_name='quality',
    metadata=metadata
)
 
fig.update_layout(
    title='Using synthetic data to simulate corner cases in wine quality'
)
fig.show()


predictions2 = forest.predict(X_validation)
print(accuracy_score(Y_validation, predictions2))
 
X_synth = synthetic_data.values[:,0:11]
y_synth = synthetic_data.values[:,11]
predictions4 = forest.predict(X_synth)
 
# Evaluate predictions
print(accuracy_score(y_synth, predictions4))
print(confusion_matrix(y_synth, predictions4))
print(classification_report(y_synth, predictions4))

