from ._utils import *
from ...ensemble import Ensemble
from ...ensemble import predict_proba

def prune_greedy_exact(
    H: Ensemble,
    X,
    y,
    k: int,
    w = None,
    voting: str = "hard"
):
    ne, w = get_greedy_params(H, w)
    proba = predict_proba(
        H, X, voting=voting, squeeze=False)
    
    # Change y to one-hot encoding
    y_one_hot = np.eye(proba.shape[-2])[y]
    
    # Calculate the scores
    scores = [np.mean(proba[:, :, e] - y_one_hot)**2
              for e in range(ne)]
    scores = np.array(scores)

    idx = np.argsort(scores / w)
    u = np.zeros(len(H), dtype=int)
    u[idx[:k]] = 1
    return u