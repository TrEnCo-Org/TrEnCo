import gurobipy as gp
from gurobipy import GRB

import time
import numpy as np

from .base import BasePruner
from .utils import predict_single_proba, predict

class PrunerMIP(BasePruner):
    # Create a new gurobi environment.
    env = gp.Env()
    
    # Gurobi model and variables.
    mip: gp.Model
    u: gp.tupledict[int, gp.Var]
    st: float
    
    def __init__(self, E, w=None, voting="hard", eps=1.0):
        super().__init__(E, w, voting)
        self.eps = eps
        self.wm = self.weights_.min()
        self.mip = gp.Model("MIP", env=self.env)
        self.st = 0
   
    def set_gurobi_param(self, param, value):
        # Set a Gurobi parameter.
        self.mip.setParam(param, value)  

    def prune(self, X):
        self._build_base_mip()
        self.st = 0
        return self._solve_mip(X)
        
    def reprune(self, X):
        return self._solve_mip(X)

    def _build_base_mip(self):
        ne = len(self.estimators_)
        self.u = self.mip.addVars(
            ne, vtype=GRB.BINARY, name="u")

        obj = gp.quicksum(self.u)
        self.mip.setObjective(obj, GRB.MINIMIZE)

    def _solve_mip(self, X):
        n = len(X)
        nc = self.n_classes_
        ne = len(self.estimators_)
        E = self.estimators_
        w = self.weights_
        u = self.u
        wm = self.wm
        eps = self.eps
        
        y = predict(E, X, w, self.voting_)
        p = predict_single_proba(E, X)
        
        for i in range(n):
            l = y[i]

            for k in range(nc):
                if k == l: continue
                lhs = gp.LinExpr()
                for e in range(ne):
                    lhs += (w[e]*(p[e,i,l]-p[e,i,k])*u[e])
                if k < l:
                    rhs = eps*wm
                else:
                    rhs = 0.0
                
                cons = self.mip.addLConstr(lhs >= rhs)
                cons.Lazy = 1
        
        s = time.time()
        self.mip.optimize()
        e = time.time()
        
        self.st += (e-s)

        if self.mip.status == GRB.INFEASIBLE:
            return None

        return self.asarray(u)

    @staticmethod
    def asarray(u: gp.tupledict[int, gp.Var]):
        n = len(u)
        v = np.array([u[i].X for i in range(n)])
        return (v >= 0.5).astype(int)