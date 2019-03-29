import pandas as pd
import math
import numpy as np
from operator import itemgetter
from matplotlib import pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

ESTIMATORS_NUM = 10

def transform_data(y):
    res = []
    for i in y:
        res.append('EUROPE' if i == 'EUROPE' else 'NOT EUROPE')
    return res

def initiate_data():
    data = pd.read_csv("countries.csv")
    data.fillna(0, inplace=True)

    y = data.iloc[:, 1].values
    x = data.iloc[:, 2:].values
    y = transform_data(y)
    return x, y

def iterate(prev_alphas, iter_num):
    if iter_num >= ESTIMATORS_NUM - 1:
        return prev_alphas
    cur_alphas = []
    dt = f[iter_num + 1]
    for j in range(len(y_train)):
        sample = X_train[iter_num].reshape(1, -1)
        cur_alphas.append(prev_alphas[j] * math.exp(-w[iter_num] if dt.predict(sample) == y_train[j] else w[iter_num]))
    return iterate(cur_alphas, iter_num + 1)

X, y = initiate_data()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dtc = DecisionTreeClassifier(min_samples_leaf = 2)
model = AdaBoostClassifier(n_estimators = ESTIMATORS_NUM, base_estimator = dtc, learning_rate = 2, algorithm="SAMME")

model.fit(X_train, y_train)
print(model.score(X_test,y_test))
f, w = model.estimators_, model.estimator_weights_

alphas = iterate([1 / len(y) for i in range(len(y))], 0)
alphas = np.array(alphas) / sum(alphas)

outliers_indexes = []
for i in range(len(alphas)):
    if alphas[i] > 0.05:
        outliers_indexes.append(i)

new_x = []
new_y = []
for i in range(len(X_train)):
    if i not in outliers_indexes:
        new_x.append(X_train[i])
        new_y.append(y_train[i])
model.fit(new_x, new_y)
print(model.score(X_test,y_test))
