from ...typing import *

class Feature:
    name: str
    ftype: FeatureType

    def __init__(
        self,
        name: str,
        ftype: FeatureType = FeatureType.NUMERICAL
    ):
        self.name = name
        self.ftype = ftype