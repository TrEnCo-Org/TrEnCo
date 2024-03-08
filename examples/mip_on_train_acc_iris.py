from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from trenco.pruning import prune_mip_acc

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
u = prune_mip_acc(
    rf, X_train, y_train,
    eps=1/len(X_train),
    verbose=True)

if u is not None:
    n = len(X_train)
    ne = len(rf.estimators_)
    nc = rf.n_classes_
    assert (isinstance(nc, int) and nc > 0)
    w = np.ones(ne)
    idx = [i for i in range(ne) if u[i] == 1]
    nt = len(idx)
    p = np.zeros((n, nc, ne))
    for e in range(ne):
        h = rf.estimators_[e]
        p[:, :, e] = h.predict_proba(X_train)

    y1 = np.argmax(p.dot(w), axis=1)
    y2 = np.argmax(p[:, :, idx].dot(w[idx]), axis=1)
    print(accuracy_score(y_train, y1))
    print(accuracy_score(y_train, y2))
    print(accuracy_score(y1, y2))
    
else:
    print("No pruned trees")