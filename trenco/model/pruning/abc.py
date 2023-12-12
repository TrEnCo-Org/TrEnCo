from ..ensemble import Classifier

class BasePruner:
    clf: Classifier
    pruned: Classifier | None = None
    
    def __init__(self, clf: Classifier):
        self.clf = clf

    def build(self):
        raise NotImplementedError()

    def prune(self):
        raise NotImplementedError()
