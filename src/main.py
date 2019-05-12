# FSI - Project 2
# Program to classify plants using leaf data
# We will use random forest supervised learning technique

import json
import pandas
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import *
from sklearn.ensemble import RandomForestClassifier

def generate_classifier_args(n_estimators, criterion, max_depth, min_samples_split,
                             min_samples_leaf, min_weight_fraction_leaf, max_features,
                             max_leaf_nodes, min_impurity_decrease, min_impurity_split,
                             bootstrap, oob_score, n_jobs, random_state, verbose,
                             warm_start, class_weight):

    return {'n_estimators': n_estimators, 'criterion': criterion, 'max_depth': max_depth,
            'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf,
            'min_weight_fraction_leaf': min_weight_fraction_leaf, 'max_features': max_features,
            'max_leaf_nodes': max_leaf_nodes, 'min_impurity_decrease': min_impurity_decrease,
            'min_impurity_split': min_impurity_split, 'bootstrap' : bootstrap,
            'oob_score': oob_score, 'n_jobs': n_jobs, 'random_state': random_state,
            'verbose': verbose, 'warm_start': warm_start, 'class_weight': class_weight}



#   Reading the .csv file and splitting the lines between features and labels
leaf_dataset = pandas.read_csv('../data/leaf.csv', header=None)
leaf_features = np.array(leaf_dataset.values[:,2:])
leaf_labels = np.array(list(map(int, leaf_dataset.values[:,0])))


#   Arguments for the classifiers
#   The arguments below will follow this order
#   {'n_estimators': 100, 'criterion': 'gini', 'max_depth': None,
#             'min_samples_split': 2, 'min_samples_leaf': 1,
#             'min_weight_fraction_leaf': 0.0, 'max_features': 'auto',
#             'max_leaf_nodes': None, 'min_impurity_decrease': 0.0,
#             'min_impurity_split': None, 'bootstrap' : True,
#             'oob_score': False, 'n_jobs': None, 'random_state': None,
#             'verbose': 0, 'warm_start': False, 'class_weight': None}

rf_args = []
#   min sample split
for i in range(2, 70):
    rf_args.append(generate_classifier_args(*[100, 'gini', None, i, 1, 0.0, 'auto', None, 0.0, None, True, False, None, None, 0, False, None]))

json_outfile = open('../results/json/{filename}.json'.format(filename='min_sample_split'), 'w+', encoding='UTF-8')

#   Creating the 10-folds for cross validation

kf = KFold(n_splits=10, shuffle=True, random_state=13785)

fold_index = 1
for train_index, test_index in kf.split(leaf_features):
    json_obj = {'fold{index}'.format(index=fold_index): []}
    features_train, features_test = leaf_features[train_index], leaf_features[test_index]
    labels_train, labels_test = leaf_labels[train_index], leaf_labels[test_index]

    for args in rf_args:
        classifier = RandomForestClassifier(**args)
        classifier.fit(features_train, labels_train)
        predicts = classifier.predict(features_test)

        json_obj['fold{index}'.format(index=fold_index)]\
            .append({'args': args,
                     'metrics': {
                         'accuracy': accuracy_score(labels_test, predicts),
                         'f1_score': {
                             'micro': f1_score(labels_test, predicts, average='micro'),
                             'macro': f1_score(labels_test, predicts, average='macro'),
                             'weighted': f1_score(labels_test, predicts, average='weighted')
                         },
                         'precision': {
                             'micro': precision_score(labels_test, predicts, average='micro'),
                             'macro': precision_score(labels_test, predicts, average='macro'),
                             'weighted': precision_score(labels_test, predicts, average='weighted')
                         }
                     }
                    })
    json_outfile.write(json.dumps(json_obj))
    fold_index += 1

json_outfile.close()