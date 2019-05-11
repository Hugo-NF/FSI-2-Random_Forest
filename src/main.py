# FSI - Project 2
# Program to classify plants using leaf data
# We will use random forest supervised learning technique

import pandas
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

leaf_dataset = pandas.read_csv('data/leaf.csv', header= None)
leaf_features = leaf_dataset.values[:, 2:]
leaf_labels = list(map(int, leaf_dataset.values[:,0]))


