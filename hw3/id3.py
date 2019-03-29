import numpy as np
import pandas as pd
from pprint import pprint
from operator import itemgetter
import math
import random
import sys
impossibru_threshold = 1.23456789

def partition(a, thresh):
    if (thresh == impossibru_threshold):
        return {"= {}".format(c): (a == c).nonzero()[0] for c in np.unique(a)}
    else:
        return {'>= {}'.format(thresh): (a >= thresh).nonzero()[0],
                '< {}'.format(thresh): (a < thresh).nonzero()[0]}

def entropy(s):
    res = 0
    val, counts = np.unique(s, return_counts=True)
    freqs = counts.astype('float') / len(s)
    for p in freqs:
        if p != 0.0:
            res -= p * np.log2(p)
    return res


def is_num(a):
    return type(a[0]) == float or type(a[0]) == int or type(a[0]) == np.float64


def is_str(a):
    return type(a[0]) == str


def ig(y, x):
    x_fixed, y_fixed, count_unknown = fix(y, x)   #correct

    info_gain = 0
    thresh = impossibru_threshold

    if is_num(x):
        info_gain, thresh = information_gain_for_real(y_fixed, x_fixed)
    else:
        info_gain = information_gain(y_fixed, x_fixed)

    if entropy(x_fixed) == 0:
        return 0, thresh
    else:
        info_gain *= (len(x_fixed) - count_unknown) / (len(x_fixed) * entropy(x_fixed))
        return info_gain, thresh

def fix(y, x):
    x_res, y_res, count_unknown = [], [], 0
    if is_num(x):
        mean = np.mean(
            np.array([
                not np.isnan(x_item) and x_item for x_item in x
            ])
        )
        for i in range(len(x)):
            if np.isnan(x[i]):
                x[i] = mean
                count_unknown += 1

        inds = np.array(x).argsort()
        x_res = x[inds]
        y_res = y[inds]
    elif is_str(x):
        x_without_nan = np.array([
            not ((type(x_item) == int or type(x_item) == float) and math.isnan(x_item)) and x_item for x_item in x
        ])
        val, counts = np.unique(x_without_nan, return_counts=True)
        # index = (-counts).argsort()[:len(counts)][0]
        # common_value = val[index]
        for i in range(len(x)):
            if (type(x[i]) == int or type(x[i]) == float) and math.isnan(x[i]):
                x[i] = val[random.randint(0,len(val)-1)]
                count_unknown += 1
        x_res = x
        y_res = y
    else:
        x_res = x
        y_res = y

    return x_res, y_res, count_unknown


def information_gain(y, x):
    res = entropy(y)

    # We partition x, according to attribute values x_i
    val, counts = np.unique(x, return_counts=True)
    freqs = counts.astype('float') / len(x)

    # We calculate a weighted average of the entropy
    for p, v in zip(freqs, val):
        res -= p * entropy(y[x == v])
    return res


def information_gain_for_real(y, x):
    splits = []
    for i in range(len(x) - 1):
        splits.append((x[i] + x[i + 1]) / 2.0)

    splits = np.unique(splits)
    max_ig = 0
    thresh = 0
    for split in splits:
        x_splitted = np.array([
            (x_item < split) * 1.0
            for x_item in x
        ])
        if information_gain(y, x_splitted) > max_ig:
            max_ig = information_gain(y, x_splitted)
            thresh = split
    return max_ig, thresh


def is_pure(s):
    return len(set(s)) == 1


def predict(x, tree):
    if type(tree) is not dict:
        res = {}
        val, counts = np.unique(tree, return_counts=True)
        freqs = counts.astype('float') / len(tree)
        for p, v in zip(freqs, val):
            res[v] = p
        res = sorted(res.items(), key=itemgetter(1), reverse=True)
        # return "Chance of answer to be {} is {}".format(res[0][0], res[0][1])
        return res[0][0]
    field = list(tree.keys())[0].split()[0]
    operator = list(tree.keys())[0].split()[1]
    value = list(tree.keys())[0].split()[2]
    if (type(x[field]) == int or type(x[field]) == float or type(x[field]) == np.float64) and math.isnan(x[field]):
        return 0.0
    if (operator == "="):
        next_level = "{} {} {}".format(field, operator, x[field])
    else:
        if float(x[field]) >= float(value):
            next_level = "{} >= {}".format(field, value)
        else:
            next_level = "{} < {}".format(field, value)
    return predict(x, tree[next_level])


def recursive_split(x, y, fields):

    # If there could be no split, just return the original set
    if is_pure(y) or len(y) < 6:            #shrinkage by the num of leaves
        return y

    # We get attribute that gives the highest information gain
    gain = np.array([
        ig(y, x_attr)
        for x_attr in x.T
    ])
    selected_attr = np.argmax(gain[:, 0], axis=0)
    # If there's no gain at all, nothing has to be done, just return the original set
    if np.all(gain[:,0] < 1e-6):
        return y

    # We split using the selected attribute
    sets = partition(x[:, selected_attr], gain[selected_attr, 1])

    res = {}
    for k, v in sets.items():
        y_subset = y.take(v, axis=0)
        x_subset = x.take(v, axis=0)
        res["{} {}".format(fields[selected_attr], k)] = recursive_split(
            x_subset, y_subset, fields)

    return res


pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

train_set = pd.read_csv('titanic_modified.csv')
# train_set = pd.read_csv('tennis.csv')
X = train_set.iloc[:, :6].values
y = train_set.iloc[:, 6].values

fields = list(train_set.columns.values)
tree = recursive_split(X, y, fields)
logFile=open('out.txt', 'w')
pprint(tree, logFile)
x_ = train_set.iloc[:, :6]
test_res = []
for _, row in x_.iterrows():
    test_res.append(float(predict(row, tree)))
print(np.mean(test_res == y))