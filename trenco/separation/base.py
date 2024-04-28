from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np

from ..feature import FeatureEncoder

class BaseSeparator(ABC):
    def __init__(
        self,
        E,
        w,
        fe: FeatureEncoder,
        voting="hard"
    ):
        self.estimators_ = deepcopy(E)
        self.weights_ = np.array(w)
        self.feature_encoder_ = fe
        self.voting_ = voting
    
    
    @abstractmethod
    def separate(self, X):
        raise NotImplementedError("Method separate not implemented.")