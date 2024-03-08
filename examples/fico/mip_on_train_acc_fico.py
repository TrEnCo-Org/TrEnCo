from pathlib import Path

# scikit-learn:
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

from trenco.pruning import prune_mip_acc
from trenco.ensemble import predict

folder = Path(__file__).parent 
data_folder = folder
data_path = str(data_folder / 'FICO.full.csv')
df = pd.read_csv(data_path)
y = df['Class'].values
y = np.array(y)
X = df.drop(columns=['Class']).values

# Random shuffle
n = X.shape[0]
idx = np.arange(n)
np.random.shuffle(idx)
m = 200
idx = idx[:m]
X, y = X[idx], y[idx]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=0
)
rf.fit(X_train, y_train)

u = prune_mip_acc(rf, X_train, y_train, verbose=True)
if u is not None:
    n = len(X_train)
    ne = len(rf)
    nc = rf.n_classes_
    assert (isinstance(nc, int) and nc > 0)
    w = np.ones(ne)
    nt = np.sum(u)
    
    y1 = predict(rf, X_train, w)
    y2 = predict(rf, X_train, w*u)

    print(f"Number of trees: {ne}")
    print(f"Number of trees in the pruned ensemble: {nt}")
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