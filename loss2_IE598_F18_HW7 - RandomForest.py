# coding: utf-8

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

wine = pd.read_csv('https://archive.ics.uci.edu/'
'ml/machine-learning-databases/'
'wine/wine.data', header=None)

wine.columns = ['Class label', 'Alcohol',
'Malic acid', 'Ash',
'Alcalinity of ash', 'Magnesium',
'Total phenols', 'Flavanoids',
'Nonflavanoid phenols',
'Proanthocyanins',
'Color intensity', 'Hue',
'OD280/OD315 of diluted wines',
'Proline']


X, y = wine.iloc[:, 1:].values, wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

# Part 1: Random Forest Estimators
forest = RandomForestClassifier(n_estimators=25, random_state=1, n_jobs=-1)
forest.fit(X_train, y_train)
scores = cross_val_score(estimator=forest, X=X_train, y=y_train, cv=10, n_jobs=-1)
print('CV1 accuracy score: %s' % scores)
print('CV1 accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
tscores = cross_val_score(estimator=forest, X=X_test, y=y_test, cv=5, scoring='accuracy', n_jobs=-1)
print('CV1 avg. test score: %.4f' % (np.mean(tscores)), '\n')

forest = RandomForestClassifier(n_estimators=100, random_state=1, n_jobs=-1)
forest.fit(X_train, y_train)
scores = cross_val_score(estimator=forest, X=X_train, y=y_train, cv=10, n_jobs=-1)
print('CV2 accuracy score: %s' % scores)
print('CV2 accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
tscores = cross_val_score(estimator=forest, X=X_test, y=y_test, cv=5, scoring='accuracy', n_jobs=-1)
print('CV2 avg. test score: %.4f' % (np.mean(tscores)), '\n')

forest = RandomForestClassifier(n_estimators=500, random_state=1, n_jobs=-1)
forest.fit(X_train, y_train)
scores = cross_val_score(estimator=forest, X=X_train, y=y_train, cv=10, n_jobs=-1)
print('CV3 accuracy score: %s' % scores)
print('CV3 accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
tscores = cross_val_score(estimator=forest, X=X_test, y=y_test, cv=5, scoring='accuracy', n_jobs=-1)
print('CV3 avg. test score: %.4f' % (np.mean(tscores)), '\n')

forest = RandomForestClassifier(n_estimators=1000, random_state=1, n_jobs=-1)
forest.fit(X_train, y_train)
scores = cross_val_score(estimator=forest, X=X_train, y=y_train, cv=10, n_jobs=-1)
print('CV4 accuracy score: %s' % scores)
print('CV4 accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
tscores = cross_val_score(estimator=forest, X=X_test, y=y_test, cv=5, scoring='accuracy', n_jobs=-1)
print('CV4 avg. test score: %.4f' % (np.mean(tscores)), '\n')

forest = RandomForestClassifier(n_estimators=10000, random_state=1, n_jobs=-1)
forest.fit(X_train, y_train)
scores = cross_val_score(estimator=forest, X=X_train, y=y_train, cv=10, n_jobs=-1)
print('CV5 accuracy score: %s' % scores)
print('CV5 accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
tscores = cross_val_score(estimator=forest, X=X_test, y=y_test, cv=5, scoring='accuracy', n_jobs=-1)
print('CV5 avg. test score: %.4f' % (np.mean(tscores)), '\n')

print(" ")
print('Best model: CV4, n_estimators = 1000')
print('CV4 - Individual Feature Importance:')


# Part 2: Random Forest - Feature Importance
feat_labels = wine.columns[1:]
forest = RandomForestClassifier(n_estimators=1000, random_state=1, n_jobs=-1)
forest.fit(X_train, y_train)

importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))
plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]), importances[indices], align='center')

plt.xticks(range(X_train.shape[1]), feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()

#######################################################################################################################
print()
print("My name is Joseph Loss")
print("My NetID is: loss2")
print("I hereby certify that I have read the University policy on Academic Integrity"
" and that I am not in violation.")