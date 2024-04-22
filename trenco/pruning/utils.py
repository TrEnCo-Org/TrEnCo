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

def predict_single_proba(E, X):    
    if X.ndim == 1:
        # If the dataset X is 1D,
        # reshape it to 2D so that
        # it can be fed to the estimators.
        X = X.reshape(1, -1)
   
    # Predicted probabilities:
    # for each estimator e on the 
    # dataset X.
    p = [e.predict_proba(X) for e in E]
    
    # Stack the arrays
    p = np.stack(p)
    assert(p.ndim == 3)
    
    # The result is a 3D array (ne, n, nc)
    # where ne is the number of estimators,
    # n is the number of samples in the dataset X,
    # and nc is the number of classes.
    return p

def predict_proba(E, X, w, voting = "hard"):
    # Check if the voting method is supported.
    if voting not in ["soft", "hard"]:
        raise ValueError("Unsupported voting method. voting must be 'soft' or 'hard'")

    # Predict the probabilities
    # for each sample,
    # each class, and each estimator.
    p = predict_single_proba(E, X)

    if voting == "hard":
        # If the voting method is hard,
        # convert the probabilities to
        # one-hot encoded predictions.
        p = to_one_hot(p)

    # Swap the axes such that the result
    # is a 3D array (n, nc, ne):
    # (ne, n, nc) -> (n, ne, nc)
    p = np.moveaxis(p, [0, 1, 2], [2, 0, 1])

    # Multiply the probabilities by the weights
    p = p.dot(w)
    return p

def predict(E, X, w, voting = "hard"):
    # Predicted class:
    # for each sample x in the dataset X.
    p = predict_proba(E, X, w, voting=voting)
    l = np.argmax(p, axis=-1)
    return l