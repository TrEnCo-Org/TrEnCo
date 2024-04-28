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

    # Probability variables.
    vote_vars: gp.tupledict[int, gp.Var]

    # Consistency variables.
    # Numerical features.
    num_vars: gp.tupledict[tuple[str, int], gp.Var]
    # Binary features.
    bin_vars: gp.tupledict[str, gp.Var]
    # Categorical features.
    cat_vars: gp.tupledict[str, gp.Var]

    # Separation variables.
    vote_u_vars: gp.tupledict[int, gp.Var]
    
    # Constraints:

    # Tree path constraints.
    root_conss: gp.tupledict[int, gp.Constr]
    flow_conss: gp.tupledict[tuple[int, int], gp.Constr]
    branch_to_left_conss: gp.tupledict[tuple[int, int], gp.Constr]
    branch_to_right_conss: gp.tupledict[tuple[int, int], gp.Constr]
    
    # Feature consistency constraints.
    # Numerical features.
    num_start_conss: gp.tupledict[str, gp.Constr]
    num_level_conss: gp.tupledict[tuple[str, int], gp.Constr]
    num_left_conss: gp.tupledict[tuple[str, int, int, int], gp.Constr]
    num_right_conss: gp.tupledict[tuple[str, int, int, int], gp.Constr]
    num_right_eps_conss: gp.tupledict[tuple[str, int, int, int], gp.Constr]
    # Binary features.
    bin_left_conss: gp.tupledict[tuple[str, int, int], gp.Constr]
    bin_right_conss: gp.tupledict[tuple[str, int, int], gp.Constr]
    # Categorical features.
    cat_left_conss: gp.tupledict[tuple[str, int, int], gp.Constr]
    cat_right_conss: gp.tupledict[tuple[str, int, int], gp.Constr]
    cat_sum_conss: gp.tupledict[str, gp.Constr]
    
    vote_conss: gp.tupledict[int, gp.Constr]
    vote_u_conss: gp.tupledict[int, gp.Constr]
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
        self.build_base_mip()

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
                self.reset_mip()
        return res

    def separate_classes(
        self,
        u,
        cl: int,
        cp: int
    ):  
        self.add_vote_u_vars([cl, cp])
        self.add_vote_u_conss(u, [cl, cp])
        self.add_maj_class_conss(cl)
        self.add_objective(cl, cp)

        self.mip.optimize()
        
        if self.mip.status == GRB.INFEASIBLE:
            return None

        x = self.get_sol()
        obj = self.mip.ObjVal
        return obj, x

    def set_gurobi_param(self, param, value):
        self.mip.setParam(param, value)

    def build_base_mip(self):
        # Variables:
        # self.add_x_vars()
        self.add_flow_vars()
        self.add_branch_vars()
        self.add_vote_vars()
        self.add_num_vars()
        self.add_bin_vars()
        self.add_cat_vars()

        # Constraints:

        # Path and flow constraints.
        self.add_root_conss()
        self.add_flow_conss()
        self.add_branch_to_left_conss()
        self.add_branch_to_right_conss()

        # Feature consistency constraints.
        self.add_num_conss()
        self.add_bin_conss()
        self.add_cat_conss()

        # Vote constraints.
        self.add_vote_conss()

    def reset_mip(self):
        self.mip.remove(self.vote_u_vars)
        self.mip.remove(self.vote_u_conss)
        self.mip.remove(self.maj_class_conss)

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

    def add_flow_vars(self):
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
            (f, j)
            for f in em.levels
            for j in range(len(em.levels[f]))
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
            f for f in fe.types
            if fe.types[f] == Feature.BINARY
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
        idx = [v for v in fe.inv_cat]

        if len(idx) == 0:
            self.cat_vars = gp.tupledict()
            return

        self.cat_vars = self.mip.addVars(
            idx,
            vtype=GRB.BINARY,
            name="nu"
        )

    def add_vote_vars(self):
        em = self.tree_ensemble_manager_
        self.vote_vars = self.mip.addVars(
            em.n_classes,
            lb=0.0,
            vtype=GRB.CONTINUOUS,
            name="z"
        )

    def add_vote_u_vars(self, classes: list[int]):
        self.vote_u_vars = self.mip.addVars(
            classes,
            lb=0.0,
            vtype=GRB.CONTINUOUS,
            name="zeta"
        )

    def add_root_conss(self):
        em = self.tree_ensemble_manager_
        self.root_conss = self.mip.addConstrs(
            self.flow_vars[(t, em[t].root)] == 1
            for t in range(em.n_trees)
        )

    def add_flow_conss(self):
        em = self.tree_ensemble_manager_
        self.flow_conss = self.mip.addConstrs(
            self.flow_vars[(t, n)] ==
            self.flow_vars[(t, em[t].left[n])]
            + self.flow_vars[(t, em[t].right[n])]
            for t in range(em.n_trees)
            for n in em[t].inner_nodes
        )

    def add_branch_to_left_conss(self):
        em = self.tree_ensemble_manager_
        self.branch_to_left_conss = self.mip.addConstrs(
            gp.quicksum(
                self.flow_vars[(t, em[t].left[n])]
                for n in em[t].nodes_at_depth(d)
            ) <= self.branch_vars[(t, d)]
            for t in range(em.n_trees)
            for d in range(em[t].max_depth)
        )

    def add_branch_to_right_conss(self):
        em = self.tree_ensemble_manager_
        self.branch_to_right_conss = self.mip.addConstrs(
            gp.quicksum(
                self.flow_vars[(t, em[t].right[n])]
                for n in em[t].nodes_at_depth(d)
            ) <= 1 - self.branch_vars[(t, d)]
            for t in range(em.n_trees)
            for d in range(em[t].max_depth)
        )

    def add_num_conss(self):
        self.add_num_start_conss()
        self.add_num_level_conss()
        self.add_num_left_conss()
        self.add_num_right_conss()
        self.add_num_right_eps_conss()

    def add_num_start_conss(self):
        em = self.tree_ensemble_manager_
        self.num_start_conss = self.mip.addConstrs(
            self.num_vars[(f, 0)] == 1.0
            for f in em.levels
        )

    def add_num_level_conss(self):
        em = self.tree_ensemble_manager_
        self.num_level_conss = self.mip.addConstrs(
            self.num_vars[(f, j-1)] >=
            self.num_vars[(f, j)]
            for f in em.levels
            for j in range(1, len(em.levels[f]))
        )

    def add_num_left_conss(self):
        em = self.tree_ensemble_manager_
        self.num_left_conss = self.mip.addConstrs(
            self.num_vars[(f, j)] <=
            1 - self.flow_vars[(t, em[t].left[n])]
            for f in em.levels
            for j in range(1, len(em.levels[f]))
            for t in range(em.n_trees)
            for n in em[t].nodes_split_on(f)
            if em[t].threshold[n] == em.levels[f][j]
        )
            
    def add_num_right_conss(self):
        em = self.tree_ensemble_manager_
        self.num_right_conss = self.mip.addConstrs(
            self.num_vars[(f, j-1)] >=
            self.flow_vars[(t, em[t].right[n])]
            for f in em.levels
            for j in range(1, len(em.levels[f]))
            for t in range(em.n_trees)
            for n in em[t].nodes_split_on(f)
            if em[t].threshold[n] == em.levels[f][j]
        )
    
    def add_num_right_eps_conss(self):
        em = self.tree_ensemble_manager_
        eps = em.eps
        self.num_right_eps_conss = self.mip.addConstrs(
            self.num_vars[(f, j)] >=
            eps[f]*self.flow_vars[(t, em[t].right[n])]
            for f in em.levels
            for j in range(1, len(em.levels[f]))
            for t in range(em.n_trees)
            for n in em[t].nodes_split_on(f)
            if em[t].threshold[n] == em.levels[f][j]
        )

    def add_bin_conss(self):
        self.add_bin_left_conss()
        self.add_bin_right_conss()

    def add_bin_left_conss(self):
        fe = self.feature_encoder_
        em = self.tree_ensemble_manager_
        self.bin_left_conss = self.mip.addConstrs(
            self.bin_vars[f] <=
            1 - self.flow_vars[(t, em[t].left[n])]
            for f in fe.types
            for t in range(em.n_trees)
            if fe.types[f] == Feature.BINARY
            for n in em[t].nodes_split_on(f)
        )

    def add_bin_right_conss(self):
        fe = self.feature_encoder_
        em = self.tree_ensemble_manager_
        self.bin_right_conss = self.mip.addConstrs(
            self.bin_vars[f] >=
            self.flow_vars[(t, em[t].right[n])]
            for f in fe.types
            for t in range(em.n_trees)
            if fe.types[f] == Feature.BINARY
            for n in em[t].nodes_split_on(f)
        )

    def add_cat_conss(self):
        self.add_cat_left_conss()
        self.add_cat_right_conss()
        self.add_cat_sum_conss()

    def add_cat_left_conss(self):
        fe = self.feature_encoder_
        em = self.tree_ensemble_manager_
        self.cat_left_conss = self.mip.addConstrs(
            self.cat_vars[c] <=
            self.flow_vars[(t, em[t].left[n])]
            for c in fe.inv_cat
            for t in range(em.n_trees)
            for n in em[t].nodes_split_on(fe.inv_cat[c])
            if em[t].category[n] == c
        )

    def add_cat_right_conss(self):
        fe = self.feature_encoder_
        em = self.tree_ensemble_manager_
        self.cat_right_conss = self.mip.addConstrs(
            self.cat_vars[c] >=
            self.flow_vars[(t, em[t].right[n])]
            for c in fe.inv_cat
            for t in range(em.n_trees)
            for n in em[t].nodes_split_on(fe.inv_cat[c])
            if em[t].category[n] == c
        )

    def add_cat_sum_conss(self):
        fe = self.feature_encoder_
        self.cat_sum_conss = self.mip.addConstrs(
            gp.quicksum(
                self.cat_vars[c]
                for c in fe.cat[f]
            ) == 1.0
            for f in fe.cat
        )

    def add_vote_conss(self):
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
        self.vote_conss = self.mip.addConstrs(
            self.vote_vars[c] == rhs[c]
            for c in range(em.n_classes)
        )
    
    def add_vote_u_conss(
        self,
        u,
        classes: list[int],
    ):
        em = self.tree_ensemble_manager_
        w = self.weights_
        rhs = gp.tupledict()
        for c in classes:
            rhs[c] = gp.LinExpr()
            for t, tm in enumerate(em):
                for n in tm.leaves:
                    vs = tm.tree.value[n][0]
                    if self.voting_ == "hard":
                        mv = np.max(vs)
                        vs = (vs == mv).astype(int)
                    v = vs[c]
                    rhs[c] += w[t]*v*u[t]*self.flow_vars[(t, n)]
        self.vote_u_conss = self.mip.addConstrs(
            self.vote_u_vars[c] == rhs[c]
            for c in classes
        )

    def add_maj_class_conss(self, mc: int):
        em = self.tree_ensemble_manager_
        w = self.weights_
        wm = w.min()
        eps = self.eps
        rhs = gp.tupledict()
        for c in range(em.n_classes):
            if c == mc:
                continue
            elif c < mc:
                rhs[c] = eps * wm
            else:
                rhs[c] = 0.0
        self.maj_class_conss = self.mip.addConstrs(
            self.vote_vars[mc]-self.vote_vars[c] >= rhs[c]
            for c in range(em.n_classes)
            if c != mc
        )
            
    def add_objective(self, cl, cp):
        self.mip.setObjective(
            self.vote_u_vars[cp] - self.vote_u_vars[cl],
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