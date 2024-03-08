from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

Ensemble = AdaBoostClassifier \
    | BaggingClassifier \
    | GradientBoostingClassifier \
    | RandomForestClassifier # \
    # | VotingClassifier
