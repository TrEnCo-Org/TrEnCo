from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from trenco.pruning import prune_mip_exact
from trenco.ensemble import predict

import numpy as np

# Load the iris dataset
X, y = load_iris(return_X_y=True)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Train a Random Forest
rf = RandomForestClassifier(
    n_estimators=100, random_state=42)

# Fit the Random Forest
rf.fit(X_train, y_train)

# Prune the Random Forest
u = prune_mip_exact(
    rf, X_train,
    eps=1/len(X_train),
    verbose=True)

if u is not None:
    n = len(X_train)
    ne = len(rf.estimators_)
    nc = rf.n_classes_
    assert (isinstance(nc, int) and nc > 0)
    w = np.ones(ne)
    
    y1 = predict(rf, X_train, w)
    y2 = predict(rf, X_train, w*u)
    print(f"Number of trees: {ne}")
    print(f"Number of pruned trees: {ne-np.sum(u)}")
    print(f"Accuracy of the original ensemble: {accuracy_score(y_train, y1)}")
    print(f"Accuracy of the pruned ensemble: {accuracy_score(y_train, y2)}")
    print(f"Loyalty of the pruned ensemble: {accuracy_score(y1, y2)}")
    
    y1 = predict(rf, X_test, w)
    y2 = predict(rf, X_test, w*u)
    
    print(f"Accuracy of the original ensemble on test set: {accuracy_score(y_test, y1)}")
    print(f"Accuracy of the pruned ensemble on test set: {accuracy_score(y_test, y2)}")
    print(f"Loyalty of the pruned ensemble on test set: {accuracy_score(y1, y2)}")
    
else:
    print("No pruned trees")