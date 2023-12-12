from enum import Enum

class FeatureType(Enum):
    NUMERICAL = 'F'
    BINARY = 'B'
    CATEGORICAL = 'C'
    DISCRETE = 'D'

    def numerical(self) -> bool:
        return self in [
            FeatureType.NUMERICAL,
            FeatureType.BINARY,
            FeatureType.DISCRETE
        ]

    def discrete(self) -> bool:
        return self == FeatureType.DISCRETE

    def categorical(self) -> bool:
        return self == FeatureType.CATEGORICAL

    def binary(self) -> bool:
        return self == FeatureType.BINARY

    def scalable(self) -> bool:
        return self in [
            FeatureType.NUMERICAL,
            FeatureType.DISCRETE
        ]