import numpy as np

def get_greedy_params(H, X, w):
    ne = len(H)
    
    if w is None:
        w = np.ones(ne)
    else:
        w = np.array(w)
    assert (len(w) == ne)
    
    return ne, w