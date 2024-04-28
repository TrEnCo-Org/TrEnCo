import itertools

import numpy as np

import gurobipy as gp
from gurobipy import GRB

from .base import BaseSeparator
from ..feature import (
    Feature,
    FeatureEncoder
)
from ..tree import TreeEnsembleManager

class SeparatorMIP(BaseSeparator):
    # Gurobi fields:

    # Gurobi environment as static field.
    env = gp.Env()
    
    # MIP model.
    mip: gp.Model
    
    # Variables:
    
    # Sample variables.
    x: gp.tupledict[str, gp.Var]
    
    # Node variables.
    flow_vars: gp.tupledict[tuple[int, int], gp.Var]

    # Flow variables.
    branch_vars: gp.tupledict[tuple[int, int], gp.Var]

    # Consistency variables.
    # Numerical features.
    num_vars: gp.tupledict[tuple[str, int], gp.Var]
    # Binary features.
    bin_vars: gp.tupledict[str, gp.Var]
    # Categorical features.
    cat_vars: gp.tupledict[str, gp.Var]

    # Score variables.
    prob_vars: gp.tupledict[int, gp.Var]
    prob_u_vars: gp.tupledict[int, gp.Var]
    
    # Constraints:
    prob_conss: gp.tupledict[int, gp.Constr]
    prob_u_conss: gp.tupledict[int, gp.Constr]
    maj_class_conss: gp.tupledict[int, gp.Constr]

    def __init__(
        self,
        E,
        w,
        fe: FeatureEncoder,
        voting="hard",
        eps=1.0
    ):
        super().__init__(E, w, fe, voting=voting)
        self.tree_ensemble_manager_ = TreeEnsembleManager(E, fe)
        self.eps = eps

        # Create the MIP model.        
        self.mip = gp.Model("SEP_MIP", env=self.env)
 
    def separate(self, u) -> list[dict[str, float]]:
        em = self.tree_ensemble_manager_
        res = []
        for cl in range(em.n_classes):
            for cp in range(em.n_classes):
                if cl == cp:
                    continue
                r = self.separate_classes(u, cl, cp)
                if r is not None:
                    obj, x = r
                    if cp < cl and obj >= 0.0:
                        res.append(x)
                    elif cp > cl and obj > 0.0:
                        res.append(x)
        return res

    def separate_classes(
        self,
        u,
        cl: int,
        cp: int
    ):
        self.build_base_mip()
        
        self.add_prob_u_vars([cl, cp])
        self.add_prob_u_conss(u, [cl, cp])
        self.add_maj_class_conss(cl)
        self.add_objective(cl, cp)

        self.mip.optimize()
        
        if self.mip.status == GRB.INFEASIBLE:
            self.clear_mip()
            return None

        x = self.get_sol()
        obj = self.prob_u_vars[cp].X - self.prob_u_vars[cl].X
        self.clear_mip()
        return obj, x

    def set_gurobi_param(self, param, value):
        self.mip.setParam(param, value)

    def clear_mip(self):
        self.mip.remove(self.prob_u_vars)
        self.mip.remove(self.prob_u_conss)
        self.mip.remove(self.maj_class_conss)

    def build_base_mip(self):
        # Variables:
        # self.add_x_vars()
        self.add_path_vars()
        self.add_branch_vars()
        self.add_num_vars()
        self.add_bin_vars()
        self.add_cat_vars()
        self.add_prob_vars()

        # Constraints:

        # Path and flow constraints.
        self.add_root_conss()
        self.add_children_conss()
        self.add_depth_to_left_conss()
        self.add_depth_to_right_conss()

        # Score constraints.
        self.add_prob_conss()

        # Feature consistency constraints.
        self.add_num_conss()
        self.add_bin_conss()
        self.add_cat_conss()

    def add_x_vars(self):
        # Variables for each feature.
        # TODO: need to be fixed.
        fe = self.feature_encoder_
        self.x = self.mip.addVars(
            fe.features,
            lb=0.0,
            ub=1.0,
            vtype=GRB.CONTINUOUS,
            name="x"
        )

    def add_path_vars(self):
        # Variables for each node, in each tree.
        # y_{t, n} = 1 if node n is active
        # in tree t, 0 otherwise.
        # This variable can be set as continuous.
        em = self.tree_ensemble_manager_
        idx = [
            (t, n)
            for t, tm in enumerate(em)
            for n in tm
        ]
        self.flow_vars = self.mip.addVars(
            idx,
            lb=0.0,
            ub=1.0,
            vtype=GRB.CONTINUOUS,
            name="y"
        )

    def add_branch_vars(self):  
        # Indicator variables for each depth,
        # in each tree.
        # lam_{t, d} = 1 if the path goes to
        # left branch at depth d in tree t,
        # 0 otherwise.
        em = self.tree_ensemble_manager_
        idx = [
            (t, d)
            for t, tm in enumerate(em)
            for d in range(tm.max_depth+1)
        ]
        self.branch_vars = self.mip.addVars(
            idx,
            vtype=GRB.BINARY,
            name="lamda"
        )

    def add_num_vars(self):
        em = self.tree_ensemble_manager_
        idx = [
            (c, j)
            for c, lvls in em.levels.items()
            for j in range(len(lvls))
        ]
        if len(idx) == 0:
            self.num_vars = gp.tupledict()
            return

        self.num_vars = self.mip.addVars(
            idx,
            lb=0.0,
            ub=1.0,
            vtype=GRB.CONTINUOUS,
            name="mu"
        )

    def add_bin_vars(self):
        fe = self.feature_encoder_
        idx = [
            c for c, ft in fe.types.items()
            if ft == Feature.BINARY
        ]
        if len(idx) == 0:
            self.bin_vars = gp.tupledict()
            return
        self.bin_vars = self.mip.addVars(
            idx,
            vtype=GRB.BINARY,
            name="om"
        )

    def add_cat_vars(self):
        fe = self.feature_encoder_
        idx = list(itertools.chain(*fe.cat.values()))
        if len(idx) == 0:
            self.cat_vars = gp.tupledict()
            return
        self.cat_vars = self.mip.addVars(
            idx,
            vtype=GRB.BINARY,
            name="nu"
        )

    def add_prob_vars(self):
        em = self.tree_ensemble_manager_
        self.prob_vars = self.mip.addVars(
            range(em.n_classes),
            lb=0.0,
            vtype=GRB.CONTINUOUS,
            name="z"
        )

    def add_prob_u_vars(self, c):
        self.prob_u_vars = self.mip.addVars(
            c,
            lb=0.0,
            vtype=GRB.CONTINUOUS,
            name="zeta"
        )

    def add_root_conss(self):
        em = self.tree_ensemble_manager_
        for t in range(em.n_trees):
            self.mip.addConstr(
                self.flow_vars[(t, 0)] == 1,
                name=f"root_{t}"
            )

    def add_children_conss(self):
        em = self.tree_ensemble_manager_
        for t, tm in enumerate(em):
            for n in tm.inner_nodes:
                l = tm.tree.children_left[n]
                r = tm.tree.children_right[n]
                assert l != r
                
                self.mip.addConstr(
                    self.flow_vars[(t, l)] + self.flow_vars[(t, r)] == self.flow_vars[(t, n)],
                    name=f"split_{t}_{n}"
                )

    def add_depth_to_left_conss(self):
        em = self.tree_ensemble_manager_
        for t, tm in enumerate(em):
            for d in range(tm.max_depth):
                lhs = gp.LinExpr()
                for n in tm.nodes_at_depth(d):
                    l = tm.tree.children_left[n]
                    assert l != -1
                    
                    lhs += self.flow_vars[(t, l)]

                rhs = self.branch_vars[(t, d)]
                self.mip.addConstr(
                    lhs <= rhs,
                    name=f"depth_left_{t}_{d}"
                )

    def add_depth_to_right_conss(self):
        em = self.tree_ensemble_manager_
        for t, tm in enumerate(em):
            for d in range(tm.max_depth):
                lhs = gp.LinExpr()
                for n in tm.nodes_at_depth(d):
                    r = tm.tree.children_right[n]
                    assert r != -1

                    lhs += self.flow_vars[(t, r)]

                rhs = 1 - self.branch_vars[(t, d)]
                self.mip.addConstr(
                    lhs <= rhs,
                    name=f"depth_right_{t}_{d}"
                )

    def add_num_conss(self):
        em = self.tree_ensemble_manager_
        for c, lvls in em.levels.items():            
            # mu_{i}^{0} = 1
            self.mip.addConstr(
                self.num_vars[(c, 0)] == 1,
                name=f"feature_consistency_numerical_{c}_0"
            )

            # mu_{i}^{j} <= mu_{i}^{j-1}
            for j in range(1, len(lvls)):
                self.mip.addConstr(
                    self.num_vars[(c, j)] <= self.num_vars[(c, j-1)],
                    name=f"feature_consistency_numerical_{c}_{j}"
                )
                lvl = lvls[j]
                for t, tm in enumerate(em):
                    for n in tm.nodes_split_on(c):
                        th = tm.threshold[n]
                        if lvl == th:
                            l = tm.tree.children_left[n]
                            r = tm.tree.children_right[n]

                            # mu_{i}^{j} <= 1 - y_{t, l}
                            self.mip.addConstr(
                                self.num_vars[(c, j)] >= 1 - self.flow_vars[(t, l)],
                                name=f"feature_consistency_left_{c}_{j}_{t}_{n}"
                            )

                            # mu_{i}^{j-1} >= y_{t, r}
                            self.mip.addConstr(
                                self.num_vars[(c, j-1)] >= self.flow_vars[(t, r)],
                                name=f"feature_consistency_right_{c}_{j}_{t}_{n}"
                            )

                            # mu_{i}^{j} >= eps_{i} * y_{t, r}
                            self.mip.addConstr(
                                self.num_vars[(c, j)] >= em.eps[c] * self.flow_vars[(t, r)],
                                name=f"feature_consistency_eps_{c}_{j}_{t}_{n}"
                            )


    def add_bin_conss(self):
        em = self.tree_ensemble_manager_
        for c in self.bin_vars:
            for t, tm in enumerate(em):
                for n in tm.nodes_split_on(c):
                    l = tm.tree.children_left[n]
                    r = tm.tree.children_right[n]

                    # om_{i} >= 1 - y_{t, l}
                    self.mip.addConstr(
                        self.bin_vars[c] >= 1 - self.flow_vars[(t, l)],
                        name=f"feature_consistency_binary_left_{c}_{t}_{n}"
                    )
                    
                    # om_{i} >= y_{t, r}
                    self.mip.addConstr(
                        self.bin_vars[c] >= self.flow_vars[(t, r)],
                        name=f"feature_consistency_binary_right_{c}_{t}_{n}"
                    )

    def add_cat_conss(self):
        fe = self.feature_encoder_
        em = self.tree_ensemble_manager_
        for c, cat in fe.cat.items():
            lhs = gp.LinExpr()
            for j in cat:
                lhs += self.cat_vars[j]
                for t, tm in enumerate(em):
                    for n in tm.nodes_split_on(c):
                        l = tm.tree.children_left[n]
                        r = tm.tree.children_right[n]
                        if tm.category[n] == j:
                            # nu_{c}^{j} >= 1 - y_{t, l}
                            self.mip.addConstr(
                                self.cat_vars[j] >= 1 - self.flow_vars[(t, l)],
                                name=f"feature_consistency_categorical_left_{c}_{j}_{t}_{n}"
                            )

                            # nu_{c}^{j} >= y_{t, r}
                            self.mip.addConstr(
                                self.cat_vars[j] >= self.flow_vars[(t, r)],
                                name=f"feature_consistency_categorical_right_{c}_{j}_{t}_{n}"
                            )
            self.mip.addConstr(
                lhs == 1.0,
                name=f"feature_consistency_categorical_{c}"
            )
    
    def add_prob_conss(self):
        em = self.tree_ensemble_manager_
        w = self.weights_
        rhs = gp.tupledict()
        for c in range(em.n_classes):
            rhs[c] = gp.LinExpr()    
            for t, tm in enumerate(em):
                for n in tm.leaves:
                    vs = tm.tree.value[n][0]
                    if self.voting_ == "hard":
                        mv = np.max(vs)
                        vs = (vs == mv).astype(int)
                    v = vs[c]
                    rhs[c] += w[t]*v*self.flow_vars[(t, n)]
        self.prob_conss = self.mip.addConstrs(
            self.prob_vars[c] == rhs[c]
            for c in range(em.n_classes)
        )
    
    def add_prob_u_conss(
        self,
        u,
        cs: list[int],
    ):
        em = self.tree_ensemble_manager_
        w = self.weights_
        rhs = gp.tupledict()
        for c in cs:
            rhs[c] = gp.LinExpr()
            for t, tm in enumerate(em):
                for n in tm.leaves:
                    vs = tm.tree.value[n][0]
                    if self.voting_ == "hard":
                        mv = np.max(vs)
                        vs = (vs == mv).astype(int)
                    v = vs[c]
                    rhs[c] += w[t]*v*u[t]*self.flow_vars[(t, n)]
        self.prob_u_conss = self.mip.addConstrs(
            self.prob_u_vars[c] == rhs[c]
            for c in cs
        )

    def add_maj_class_conss(self, cl):
        em = self.tree_ensemble_manager_
        w = self.weights_
        wm = w.min()
        eps = self.eps
        rhs = gp.tupledict()
        for c in range(em.n_classes):
            if c == cl:
                continue
            elif c < cl:
                rhs[c] = eps * wm
            else:
                rhs[c] = 0.0
        self.maj_class_conss = self.mip.addConstrs(
            self.prob_u_vars[cl] - self.prob_u_vars[c] >= rhs[c]
            for c in rhs
        )
            
    def add_objective(self, cl, cp):
        self.mip.setObjective(
            self.prob_u_vars[cp] - self.prob_u_vars[cl],
            GRB.MAXIMIZE
        )

    def get_sol(self):
        em = self.tree_ensemble_manager_
        fe = self.feature_encoder_
        sol = dict()
        for f, ft in fe.types.items():
            if ft == Feature.NUMERICAL:
                lvls = em.levels[f]
                j = 0
                while j < len(lvls) and self.num_vars[f, j].X >= 0.5:
                    j += 1
                if j == len(lvls):
                    sol[f] = lvls[-1]
                else:
                    sol[f] = lvls[j]
            elif ft == Feature.BINARY:
                sol[f] = self.bin_vars[f].X
            elif ft == Feature.CATEGORICAL:
                for j in fe.cat[f]:
                    sol[j] = self.cat_vars[j].X
        return sol