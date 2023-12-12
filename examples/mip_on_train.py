from trenco.utils.data import DatasetLoader
from trenco.model.pruning import PrunerMIP
from trenco.model.ensemble import Classifier

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

import numpy as np

# Load the dataset
folder = Path(__file__).parent 
data_folder = folder / 'resources/datasets/FICO'
data_path = str(data_folder / 'FICO.full.csv')
types_path = str(data_folder / 'FICO.featurelist.csv')
loader = DatasetLoader(data_path, types_path, "Class")

# Extract the training and test sets
train, valid, test = loader.split(frac=0.1, ratio=[0.8, 0.1, 0.1])
X_train, y_train = train.values

# Train a random forest classifier
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train) # type: ignore

# Wrap the random forest with the trenco Classifier class
clf = Classifier(rf)

# Build and prune the classifier
# Using the MIP compressor
pruner = PrunerMIP(clf, train, valid, verbose=False)
pruner.build()
pruner.prune()
if pruner.pruned is None:
    print("No pruned classifier found.")
    exit(1)

print("Pruned classifier found.")
print("Number of trees in the original classifier: {}".format(pruner.clf.n_trees))
print("Number of trees in the pruned classifier: {}".format(pruner.pruned.n_trees))

clf_predictions_on_train = clf.predict(X_train)
pruned_predictions_on_train = pruner.pruned.predict(X_train)
true_predictions_on_train = y_train
print("Classifier accuracy on train: {}".format(np.mean(clf_predictions_on_train == true_predictions_on_train)))
print("Pruned classifier accuracy on train: {}".format(np.mean(pruned_predictions_on_train == true_predictions_on_train)))
print("Pruned classifier vs. classifier on train: {}".format(np.mean(pruned_predictions_on_train == clf_predictions_on_train)))


# Compute the accuracy of the pruned classifier
# on the test set.
X_test, y_test = test.values
clf_predictions = clf.predict(X_test)
pruned_predictions = pruner.pruned.predict(X_test)
true_predictions = y_test

clf_accuracy = np.mean(clf_predictions == true_predictions)
pruned_accuracy = np.mean(pruned_predictions == true_predictions)
pruned_vs_clf = np.mean(pruned_predictions == clf_predictions)

print("Classifier accuracy on test: {}".format(clf_accuracy))
print("Pruned classifier accuracy on test: {}".format(pruned_accuracy))
print("Pruned classifier vs. classifier on test: {}".format(pruned_vs_clf))