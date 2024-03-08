from ._utils import *
from ...ensemble import Ensemble
from ...ensemble import predict

def prune_greedy_exact(
    H: Ensemble,
    X,
    w = None,
    voting: str = "hard"
):
    ne, w = get_greedy_params(H, X, w)
    
    idx = np.arange(ne)
    np.random.shuffle(idx)
    
    u = np.ones(ne)
    y = predict(H, X, w, voting=voting)
    for i in idx:
        u[i] = 0
        yu = predict(H, X, w*u, voting=voting)
        if np.all(y == yu):
            continue
        else:
            u[i] = 1  
    return u