# FSI - Project 2
# Program to classify plants using leaf data
# We will use random forest supervised learning technique

from src.random_forest import RandomForest

rfc = RandomForest("../data/leaf.csv")

rf_args = []

#  Number of estimators ():
for i in range(1,200,5):
    rf_args.append(rfc.generate_classifier_args(*[i, 'gini', None, 2, 1, 0.0, 'auto', None, 0.0, None, True, False, None, None, 0, False, None]))

rfc.set_classifier_args(rf_args)
rfc.run_classifiers('n_estimators')
rf_args.clear()

#   Min Sample Split (The best is using 2 or 3):
for i in range(2, 70):
    rf_args.append(rfc.generate_classifier_args(*[100, 'gini', None, i, 1, 0.0, 'auto', None, 0.0, None, True, False, None, None, 0, False, None]))

rfc.set_classifier_args(rf_args)
rfc.run_classifiers('min_samples_split')
rf_args.clear()

#   Criterion (The best is using entropy)
rf_args.append(rfc.generate_classifier_args(*[100, 'gini', None, 3, 1, 0.0, 'auto', None, 0.0, None, True, False, None, None, 0, False, None]))
rf_args.append(rfc.generate_classifier_args(*[100, 'entropy', None, 3, 1, 0.0, 'auto', None, 0.0, None, True, False, None, None, 0, False, None]))

rfc.set_classifier_args(rf_args)
rfc.run_classifiers('criterion')
rf_args.clear()

#   Max Depth (Generate too much flutuation, so use none and control with another kinds of pruning):
for i in range(2,90):
    rf_args.append(rfc.generate_classifier_args(*[100, 'entropy', i, 3, 1, 0.0, 'auto', None, 0.0, None, True, False, None, None, 0, False, None]))

rfc.set_classifier_args(rf_args)
rfc.run_classifiers('max_depth')
rf_args.clear()

#   Min Samples Leaf (The best is 2 or 3)
for i in range(2,70):
    rf_args.append(rfc.generate_classifier_args(*[100, 'entropy', None, 3, i, 0.0, 'auto', None, 0.0, None, True, False, None, None, 0, False, None]))

rfc.set_classifier_args(rf_args)
rfc.run_classifiers('min_samples_leaf')
rf_args.clear()

#   Min Weight Fraction Leaf (not used)

#   Max Features (by theory the best is sqrt of samples)

#   Max Leaf Nodes (not used)

#   Min Impurity Decrease