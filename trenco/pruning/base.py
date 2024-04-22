from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np

class BasePruner(ABC):
    def __init__(self, E, w=None, voting="hard"):
        if E is None or len(E) == 0:
            raise RuntimeError("No estimators provided.")

        self.estimators_ = deepcopy(E)
        self.voting_ = voting

        if w is None:
            w = np.ones(len(E))
        elif len(w) != len(E):
            raise RuntimeError("Detected different number of weights and estimators.")
        
        self.weights_ = np.array(w)
        
        classes = [e.n_classes_ for e in E]
        if (len(set(classes)) > 1):
            raise RuntimeError("Detected different number of classes in the estimators.")
        self.classes_ = E[0].classes_
        self.n_classes_ = classes[0]
    
    @abstractmethod
    def prune(self, X):
        raise NotImplementedError("Method prune not implemented.")

    @abstractmethod
    def reprune(self, X):
        raise NotImplementedError("Method reprune not implemented.")