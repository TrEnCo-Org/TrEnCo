import pandas as pd

from ...typing import *

class Dataset:
    _data: pd.DataFrame
    _class_column: str
    _features: list[str]

    def __init__(
        self,
        data: pd.DataFrame,
        class_column: str = "Class",
    ):
        if class_column not in data.columns:
            raise ValueError(f"Class column '{class_column}' not found in data")

        self._data = data        
        self._features = self._data.columns.drop(class_column).tolist()
        self._class_column = class_column

    @property
    def values(self):
        X = self._data[self._features].values
        y = self._data[self._class_column].values
        return X, y

    def __getitem__(self, index: int) -> tuple[Sample, int]:
        row = self._data.iloc[index]
        return row[self._features], row[self._class_column]
    
    def __len__(self):
        return len(self._data)

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]