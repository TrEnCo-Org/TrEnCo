import numpy as np
import pandas as pd

from ...typing import *
from .feature import Feature
from .dataset import Dataset

class DatasetLoader:
    features: list[Feature]
    label: Feature
    
    _data: pd.DataFrame
    _types: pd.DataFrame
    _raw: pd.DataFrame
    
    def __init__(
        self,
        data_buffer: str | pd.DataFrame,
        types_buffer: str | pd.DataFrame,
        label: str = "Class",
        feature_type_index: int = 0,
        drop_single_value_features: bool = True,
        extrapolate_discrete_feature_values: bool = True
    ) -> None:
        
        if isinstance(data_buffer, str):
            self._raw = pd.read_csv(data_buffer)
        elif isinstance(data_buffer, pd.DataFrame):
            self._raw = data_buffer.copy()
        else:
            raise ValueError("Unsupported data buffer type: {}".format(type(data_buffer)))


        if isinstance(types_buffer, str):
            self._types = pd.read_csv(types_buffer)
        elif isinstance(types_buffer, pd.DataFrame):
            self._types = types_buffer.copy()
        else:
            raise ValueError("Unsupported type buffer type: {}".format(type(types_buffer)))

        if self._raw.columns.tolist() != self._types.columns.tolist():
            raise ValueError("Buffer and type buffer columns mismatch")

        if label not in self._raw.columns.tolist():
            raise ValueError("Class column not found in buffer")

        self.__extract_features(label, feature_type_index)
        if drop_single_value_features:
            self.__drop_single_value_features()
        
        self._data = self._raw[[self.label.name]].copy()
        self.__extract_numerical_feature_values()
        if extrapolate_discrete_feature_values:
            self.__extrapolate_discrete_feature_values()
        self.__extract_categorical_feature_values()

    def split(
        self,
        ratio: float | list[float] = 0.8,
        frac: float = 1.0,
        shuffle: bool = True,
        seed: int | None = None
    ) -> tuple[Dataset, ...]:
        if frac > 1.0 or frac < 0.0:
            raise ValueError("Invalid fraction value: {}".format(frac))
        
        if isinstance(ratio, float):
            if ratio >= 1.0 or ratio <= 0.0:
                raise ValueError("Invalid ratio value: {}".format(ratio))
            ratios = [ratio, 1.0 - ratio]
        elif isinstance(ratio, list):
            if min(ratio) <= 0.0 or sum(ratio) > 1.0 or sum(ratio) < 0.0:
                raise ValueError("Invalid ratio value: {}".format(ratio))
            ratios = ratio
            if sum(ratios) != 1.0:
                ratios.append(1.0 - sum(ratio))
        else:
            raise ValueError("Invalid ratio type: {}".format(type(ratio)))  

        length = int(len(self._data) * frac)
        data = self._data.iloc[:length]
        if shuffle:
            data = data.sample(frac=1.0, random_state=seed)

        datasets = []
        ratios = [0.0] + ratios
        ratios = np.cumsum(ratios)
        length = len(data)
        for i in range(len(ratios)-1):
            start = int(ratios[i] * length)
            end = int(ratios[i+1] * length)
            dataset = Dataset(data.iloc[start:end], self.label.name)
            datasets.append(dataset)
        return tuple(datasets)

    def __extract_features(self, label: str, index: int = 0):
        label_type = FeatureType(self._types[label][index])
        self.label = Feature(label, label_type)
        names = self._raw.columns.drop(self.label.name).tolist()
        types = [FeatureType(self._types[c][index]) for c in names]
        self.features = [Feature(n, t) for n, t in zip(names, types)]

    def __drop_single_value_features(self):
        features = []
        for feature in self.features:
            if len(self._raw[feature.name].unique()) > 1:
                features.append(feature)
        self.features = features

    def __extract_numerical_feature_values(self):
        for feature in self.features:
            if feature.ftype.numerical():
                self._data[feature.name] = pd.to_numeric(self._raw[feature.name], errors="raise")

    def __extract_categorical_feature_values(self):
        for feature in self.features:
            if feature.ftype.categorical():
                data = self._raw[feature.name].astype("category")
                encoded = pd.get_dummies(data, prefix=feature.name)
                encoded = encoded.astype(int)
                self._data = pd.concat([self._data, encoded], axis=1)

    def __extrapolate_discrete_feature_values(self):
        for feature in self.features:
            if feature.ftype.discrete():
                values = self._data[feature.name].values
                upper = max(values)
                lower = min(values)
                if upper == lower:
                    values = np.full(lower, values.shape)
                else:
                    values = (values - lower) / (upper - lower)
                self._data[feature.name] = values

    

    
                