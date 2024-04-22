import numpy as np

from .base import BasePruner
from .utils import predict_single_proba, predict

class PrunerGreedy(BasePruner):
    n_estimators_: int
    
    def __init__(self, E, ne, w=None, voting="hard"):
        super().__init__(E, w, voting)
        self.n_estimators_ = ne
        self.scores = np.zeros(len(self.estimators_))

    def prune(self, X):
        self.scores = np.zeros(len(self.estimators_))
        return self._prune(X)

    def reprune(self, X):
        return self._prune(X)
    
    def _prune(self, X):
        E = self.estimators_
        w = self.weights_
        ne = len(E)
        ng = self.n_estimators_
        nc = len(self.classes_)

        # Get the predictions and convert them
        # to one-hot encoded predictions.
        y = predict(E, X, w, self.voting_)
        y = np.eye(nc)[y]
        
        # Get the predicted probabilities
        # for each estimator.
        p = predict_single_proba(E, X)
        
        # Check the shapes of the arrays.
        assert y.shape == p.shape[-2:]
        
        # Compute the scores.
        scores = np.sum((p - y)**2, axis=(-2, -1))
        self.scores += scores
        
        idx = np.argsort(self.scores / w)
        u = np.zeros(ne, dtype=int)
        u[idx[:ng]] = 1
        return u
