import numpy as np

from .base import BasePruner

class PrunerRandom(BasePruner):
    n_estimators_: int
    
    def __init__(self, E, ne, seed=None):
        super().__init__(E)
        self.n_estimators_ = ne
        self.seed = seed
    
    def prune(self, X):
        np.random.seed(self.seed)
        return self._prune()
    
    def reprune(self, X):
        return self._prune()
    
    def _prune(self):
        ne = len(self.estimators_)
        nr = self.n_estimators_
        idx = np.random.choice(ne, nr, replace=False)
        u = np.zeros(ne, dtype=int)
        u[idx] = 1
        return u