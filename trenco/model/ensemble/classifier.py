from collections.abc import Iterable
from copy import deepcopy

import numpy as np
from numpy.typing import NDArray
from sklearn.tree import DecisionTreeClassifier

from ...typing import *

class Classifier:
    trees: list[DecisionTreeClassifier]
    comp: Comparator[int]
    
    def __init__(
        self,
        ensemble: Iterable[DecisionTreeClassifier],
        comp: Comparator[int] = Comparator(lambda x, y: x < y),
        weights: Iterable[float] | None = None
    ):
        self.__retrieve_tree_ensemble(ensemble)
        self.classes = np.asarray(self.trees[0].classes_)
        self.comp = comp
        self.__check_weights(weights)
        self.__prune_ensemble()

    @property
    def n_trees(self) -> int:
        return len(self.trees)

    @property
    def n_classes(self) -> int:
        return self.classes.shape[0]

    def predict_proba(self, X):
        X = np.asarray(X)
        trees = self.trees
        w = self.weights
        n = self.n_trees
        m = X.shape[0]
        k = self.n_classes
        p = np.zeros((m, n, k))
        for t, tree in enumerate(trees):
            f = tree.predict_proba(X)
            for i in range(m):
                c = np.argmax(f[i])
                p[i, t, c] = 1.0      
        return w.dot(p)

    def predict(self, X):
        X = np.asarray(X)
        n = X.shape[0]
        m = 1
        c = np.empty((n, m), dtype=int)
        p = self.predict_proba(X)
        for i in range(n):
            c[i] = self.__majority_class(p[i])
        return self.classes.take(c)

    def tree_functionals(
        self,
        sample: Sample
    ):
        trees = self.trees
        n = self.n_trees
        m = self.n_classes
        f = np.zeros((m, n))
        for t, tree in enumerate(trees):
            p = tree.predict_proba([sample])[0]
            c = np.argmax(p)
            f[c, t] = 1.0
        return f
    
    def majority_class(
        self,
        f: NDArray[np.float64]
    ) -> int:
        w = self.weights
        p = f.dot(w)
        return self.__majority_class(p)
    
    def __majority_class(
        self,
        p: NDArray[np.float64]
    ) -> int:
        key = self.comp.key
        idx = np.argwhere(p == p.max()).flatten().tolist()
        return min(idx, key=key)

    def prune(
        self,
        weights: Iterable[float],
        use_weights: bool = True
    ):
        w = np.array(weights)
        if use_weights: w = w * self.weights
        return Classifier(self.trees, self.comp, w)

    def __check_weights(
        self,
        weights: Iterable[float] | None = None
    ):
        if weights is None:
            weights = np.ones(self.n_trees)
        else:
            weights = np.asarray(weights, dtype=float)
            n = self.n_trees
            m = weights.shape[0]
            if m != n:
                raise ValueError("Number of weights must match number of trees. Got {} weights, expected {}.".format(m, n))
            if np.any(weights < 0.0):
                raise ValueError("Weights must be non-negative.")
            if np.all(np.isclose(weights, 0.0)):
                raise ValueError("At least one weight must be positive non-zero.")  
        self.weights = weights

    def __retrieve_tree_ensemble(
        self,
        ensemble: Iterable[DecisionTreeClassifier]
    ):
        trees = deepcopy(list(ensemble))
        self.trees = trees

    def __prune_ensemble(self):
        a = np.isclose(self.weights, 0.0)
        idx: list[int] = np.argwhere(a).flatten().tolist()
        self.weights = np.delete(self.weights, idx)
        for i in sorted(idx, reverse=True):
            self.trees.pop(i)