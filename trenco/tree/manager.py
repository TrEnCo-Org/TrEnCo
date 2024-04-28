from collections.abc import Iterable

import numpy as np

from ..feature import (
    Feature,
    FeatureEncoder
)

class TreeManager(Iterable[int]):
    # Fields:

    # Remove this comment.
    # if you manage to get the type
    # of the tree attribute.
    # tree: sklearn.tree._tree.Tree

    # Number of nodes in the tree.
    n_nodes: int

    # Maximum depth of the tree.
    max_depth: int

    # Set of leaf nodes.
    leaves: set[int]

    # Depth of each node.
    node_depth: dict[int, int]

    # Feature of each inner node.
    features: dict[int, str]
    
    # Threshold of each inner node.
    # Only for numerical features.
    threshold: dict[int, float]

    # Category of each inner node.
    # Only for categorical features.
    category: dict[int, str]

    def __init__(
        self,
        tree,
        fe: FeatureEncoder
    ):
        self.tree = tree
        self.n_nodes = tree.node_count
        self.max_depth = tree.max_depth
        self.leaves = set()
        self.node_depth = dict()
        self.features = dict()
        self.threshold = dict()

        stack: list[tuple[int, int]] = [(0, 0)]
        while len(stack) > 0:
            n, d = stack.pop()
            self.node_depth[n] = d        

            l = tree.children_left[n]
            r = tree.children_right[n]

            is_leaf = (l == r)
            if is_leaf:
                self.leaves.add(n)
                continue

            # Get the feature index of the node.
            fi: int = tree.feature[n]
            
            # Get the column name associated
            # with the feature index.
            c = fe.columns[fi]
            if c in fe.types.keys():
                ft = fe.types[c]
                assert (
                    ft == Feature.BINARY
                    or 
                    ft == Feature.NUMERICAL
                )
                # If the feature name is already
                # this means that the feature is
                # binary or numerical.
                self.features[n] = c
                if ft == Feature.NUMERICAL:
                    self.threshold[n] = tree.threshold[n]
            else:
                # If the feature name is not already
                # this means that the feature is
                # categorical.
                for f, cat in fe.cat.items():
                    if c in cat:
                        self.features[n] = f
                        self.category[n] = c
                        break
                else:
                    msg = f"Feature {c} not found in the feature encoder."
                    raise ValueError(msg)

            # Add the left and right children
            # to the stack. The depth of the
            # children is the depth of the
            # current node plus one.
            stack.append((l, d + 1))
            stack.append((r, d + 1))
    
    @property
    def inner_nodes(self) -> set[int]:
        nodes = set(range(self.n_nodes))
        return nodes - self.leaves

    def nodes_at_depth(
        self,
        d: int,
        only_inner: bool = True
    ) -> set[int]:
        # Return the set of nodes at depth d.
        # If only_inner is True, then only return
        # the inner nodes at depth d.
        nodes = set()
        for n, depth in self.node_depth.items():
            if depth == d:
                if only_inner and n in self.leaves:
                    continue
                nodes.add(n)
        return nodes

    def nodes_split_on(
        self,
        f: str
    ) -> set[int]:
        # Return the set of nodes that split
        # on the feature c.
        nodes = set()
        for n in self.inner_nodes:
            if self.features[n] == f:
                nodes.add(n)
        return nodes

    def __iter__(self):
        return iter(range(self.n_nodes))

    def __len__(self):
        return self.n_nodes

class TreeEnsembleManager(Iterable[TreeManager]):
    # Fields:

    # Number of trees in the ensemble.
    n_trees: int
    
    # Number of classes in the ensemble.
    n_classes: int

    # List of tree managers.
    tree_managers: list[TreeManager]

    # Levels of the numerical features.
    levels: dict[str, list[float]]
    
    # Epsilon of the numerical features.
    eps: dict[str, float]
    
    def __init__(
        self,
        E,
        fe: FeatureEncoder
    ):
        self.n_trees = len(E)
        self.n_classes = E[0].n_classes
        
        self.__fit_tree_managers(E, fe)
        self.__fit_feature_levels(fe)

    def __iter__(self):
        return iter(self.tree_managers)

    def __len__(self):
        return len(self.tree_managers)
    
    # Private methods:
    # __fit_tree_managers
    # __fit_feature_levels
    
    def __fit_tree_managers(
        self,
        E,
        fe: FeatureEncoder
    ):
        # Create a tree manager for each
        # tree in the ensemble.
        self.tree_managers = []
        for e in E:
            tm = TreeManager(e.tree_, fe)
            self.tree_managers.append(tm)

    def __fit_feature_levels(
        self,
        fe: FeatureEncoder
    ):
        self.levels = dict()
        for c, ft in fe.types.items():
            if not (ft == Feature.NUMERICAL):
                continue
            lvls = set()
            for tm in self:
                for n in tm.inner_nodes:
                    if tm.features[n] == c:
                        th = tm.threshold[n]
                        lvls.add(th)
            lvls.add(0.0)
            lvls.add(1.0)
            lvls = list(sorted(lvls))
            self.levels[c] = lvls
            self.eps[c] = self.compute_eps(lvls)

    # Static methods:
    @staticmethod
    def compute_eps(lvls: list[float]) -> float:
        diff = np.diff(lvls)
        return np.min(diff) / 2.0