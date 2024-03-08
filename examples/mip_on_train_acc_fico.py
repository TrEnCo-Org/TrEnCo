from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

from trenco.pruning import prune_mip_acc

folder = Path(__file__).parent 
data_folder = folder / 'resources/datasets/FICO'
data_path = str(data_folder / 'FICO.full.csv')
df = pd.read_csv(data_path)
y = df['Class'].values
y = np.array(y)
X = df.drop(columns=['Class']).values

# Random shuffle
n = X.shape[0]
idx = np.arange(n)
np.random.shuffle(idx)
m = 100
idx = idx[:m]
X, y = X[idx], y[idx]
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=0
)
rf.fit(X, y)

u = prune_mip_acc(rf, X, y, eps=1/m, verbose=True)
if u is not None:
    print(sum(u))
else:
    print("No pruned trees")