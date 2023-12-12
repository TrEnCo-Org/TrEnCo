import numpy as np
import gurobipy as gp

from numpy.typing import NDArray

from .abc import BasePruner
from ..ensemble.classifier import Classifier
from ...typing import *
from ...utils.data import Dataset

class BasePrunerMIP(BasePruner):
    model: gp.Model
    u: gp.tupledict[int, gp.Var]
    
    _lazy: bool = True
    _verbose: bool = False
    _eps: float = 0.5
    _threshold: float = 0.5
    _timelimit: int = 3600
    
    def __init__(
        self,
        clf: Classifier,
        **kwargs
    ):
        BasePruner.__init__(self, clf)
        self._lazy = kwargs.get("lazy", True)
        self._verbose = kwargs.get("verbose", False)
        self._threshold = kwargs.get("threshold", 0.5)
        self._timelimit = kwargs.get("timelimit", 3600)
        self._eps = 0.5 * min(self.clf.weights)

    @property
    def n_trees(self) -> int:
        return self.clf.n_trees

    @property
    def n_classes(self) -> int:
        return self.clf.n_classes

    def build(self):
        self._build_setup()
        self._build_variables()
        self._build_objective()
        self._build_all_constraints()
    
    def _build_setup(self):
        self.model = gp.Model("MIP")
        self.model.setParam("OutputFlag", self._verbose)
        self.model.setParam("LazyConstraints", self._lazy)
        self.model.setParam("TimeLimit", self._timelimit)

    def _build_variables(self):
        n = self.n_trees
        t = gp.GRB.BINARY
        self.u = self.model.addVars(n, vtype=t, name="u")

    def _build_objective(self):
        obj = gp.quicksum(self.u)
        sense = gp.GRB.MINIMIZE
        self.model.setObjective(obj, sense)

    def _build_all_constraints(self):
        raise NotImplementedError()

    def _build_pruned_classifier(self):
        w = self._pruned_weights()
        self.pruned = self.clf.prune(w, use_weights=True)

    def _pruned_weights(self):
        n = self.n_trees
        u = np.array([self.u[t].X for t in range(n)])
        w = (u >= self._threshold).astype(float)
        return w

    def _get_counter_factual(self, w: NDArray[np.float64]):
        raise NotImplementedError()

class PrunerMIP(BasePrunerMIP):
    train: Dataset
    valid: Dataset | None = None

    def __init__(
        self,
        clf: Classifier,
        train: Dataset,
        valid: Dataset | None = None,
        **kwargs
    ):
        BasePrunerMIP.__init__(self, clf, **kwargs)
        self.train = train
        self.valid = valid

    def _get_counter_factual(self, w: NDArray[np.float64]):
        if self.valid is None:
            return None
        clf = self.clf
        pruned = clf.prune(w, use_weights=True)
        for sample, klass in self.valid:
            f = clf.tree_functionals(sample)
            g = clf.majority_class(f)
            f = pruned.tree_functionals(sample)
            c = pruned.majority_class(f)
            if c != g:
                return sample, klass
        return None 

    def prune(self):
        while True:
            self.model.optimize()
            if self.model.status != gp.GRB.OPTIMAL:
                print("At some point, the model was infeasible.")
                break
            w = self._pruned_weights()
            cf = self._get_counter_factual(w)
            if cf is None:
                self._build_pruned_classifier()
                break
            else:
                sample, _ = cf
                self._build_constraints(sample)

    def _build_all_constraints(self):
        for sample, _ in self.train:
            self._build_constraints(sample)
    
    def _build_constraints(
        self,
        sample: Sample
    ):
        m = self.n_classes
        f = self.clf.tree_functionals(sample)
        g = self.clf.majority_class(f)
        for c in range(m):
            self._build_constraint(f=f, g=g, c=c)

    def _build_constraint(
        self,
        f: NDArray[np.float64],
        g: int,
        c: int
    ):
        if c == g: return
        w = np.array(self.clf.weights)
        n = self.n_trees
        p = f[g, :] - f[c, :]
        lhs = gp.LinExpr()
        for t in range(n):
            lhs += (w[t] * p[t] * self.u[t])
        if self.clf.comp(c, g):
            rhs = self._eps
        else:
            rhs = 0.0
        
        # Check that the constraint is always satisfied
        # at least by the original classifier.
        assert(sum(p * w) >= rhs)
        self.model.addConstr(lhs >= rhs)

class PrunerAccuracyMIP(BasePrunerMIP):
    pass

class PrunerLossLessMIP(BasePrunerMIP):
    pass

class PrunerLossLessDistMIP(BasePrunerMIP):
    pass   
