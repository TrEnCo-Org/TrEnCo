import numpy as np

def to_one_hot(p, axis=-1):
    # Convert the probabilities to
    # one-hot encoded predictions.
    q = np.swapaxes(p, axis, -1)
    nc = q.shape[-1]
    q = np.argmax(q, axis=-1)
    q = np.eye(nc)[q]
    p = np.swapaxes(q, axis, -1)
    return p

def predict_proba(
    H,
    X,
    voting = "hard",
    squeeze = True
):
    # Check if the voting method is supported.
    if voting not in ["soft", "hard"]:
        raise ValueError("Unsupported voting method. voting must be 'soft' or 'hard'")
    
    if squeeze and X.ndim == 1:
        # If the dataset X is 1D,
        # reshape it to 2D so that
        # it can be fed to the estimators.
        X = X.reshape(1, -1)
   
    # Predicted probabilities:
    # for each classifier e on the 
    # dataset X.
    p = [h.predict_proba(X) for h in H]
    
    # Stack the arrays
    # the result is a 3D array (ne, n, nc)
    p = np.stack(p)
    assert(p.ndim == 3)
    
    if voting == "hard":
        # If the voting method is hard,
        # convert the probabilities to
        # one-hot encoded predictions.
        p = to_one_hot(p)

    # Swap the axes such that the result
    # is a 3D array (n, nc, ne):
    # (ne, n, nc) -> (n, ne, nc)
    p = np.moveaxis(p, [0, 1, 2], [2, 0, 1])
    
    # Squeeze the array such that the result
    # is a 2D array (nc, ne) if n = 1
    if squeeze and X.shape[0] == 1:
        p = np.squeeze(p, axis=0)
    return p

def predict(H, X, w, voting = "hard"):
    # Predicted class:
    # for each sample x in the dataset X.
    p = predict_proba(H, X, voting=voting, squeeze=False)
    l = np.argmax(p.dot(w), axis=-1)
    return l