import pandas as pd
import numpy as np

from ..feature import FeatureEncoder
from .mip import PrunerMIP
from ..separation import SeparatorMIP

class PrunerFaithFull:
    def __init__(
        self,
        E,
        w,
        fe: FeatureEncoder,
        voting="hard",
        eps=1.0
    ):
        self.feature_encoder_ = fe
        self.pruner = PrunerMIP(E, w, voting, eps)
        self.separator = SeparatorMIP(E, w, fe, voting, eps)

    def prune(self):
        fe = self.feature_encoder_
        X = fe.X.values

        self.pruner.set_gurobi_param("TimeLimit", 60)
        self.separator.set_gurobi_param("TimeLimit", 60)

        while True:
            u = self.pruner.prune(X)
            u = np.array(u)
            if u.sum() == len(u):
                return u

            sols = self.separator.separate(u)
            if not sols:
                return u
        
            X = pd.DataFrame(sols, columns=fe.X.columns)