import pandas as pd
import numpy as np

from copy import deepcopy

from .typing import Feature

def ptanh(x):
    return (1.0 + np.tanh(x)) / 2.0

class FeatureEncoder:
    # Fields:
    
    # Feature types.
    types: dict[str, Feature]

    # Feature names.
    features: pd.Index[str]

    # Column names after encoding.
    columns: pd.Index[str]

    # Categories of the categorical columns.
    cat: dict[str, list[str]]
    
    def ___init__(self):
        pass
    
    def fit(self, data: pd.DataFrame):
        self.data = deepcopy(data)
        
        # clean the data
        self.__clean_data()

        # Get the columns of the data.
        self.features = self.data.columns

        # Initialize the features dictionary.
        self.types = dict()
        self.cat = dict()
        
        # Identify the feature types
        # of the columns in the data.
        
        # Start by identifying binary columns.
        # A binary column is a column with only
        # two unique values.
        # Replace the unique values with 0/1.
        # This is done to ensure that the binary
        # columns are encoded as 0/1.
        # After identifying the binary columns,
        # identify the numerical columns.
        # A numerical column is a column that is
        # not categorical and can be converted
        # to a numeric column.
        # After identifying the numerical columns,
        # identify the categorical columns.
        # A categorical column is a column that
        # is not binary or numerical.
        self.__identify_binary_features()
        self.__identify_numerical_features()
        self.__identify_categorical_features()

        # Set the data after encoding.
        self.X = self.data
        
        # Save the column names after encoding.
        self.columns = self.X.columns
        
    def transform(self):
        return self.X

    # Private methods:
    # __clean_data
    # __identify_binary_features
    # __identify_numerical_features
    # __identify_categorical_features

    def __clean_data(self):
        # Remove rows with missing values
        self.data.dropna(inplace=True)
        
        # Remove columns with only one unique value
        b = self.data.nunique() > 1
        self.data = self.data.loc[:, b]
        
    def __identify_binary_features(self):
        for c in self.features:
            # If the column has only two unique values
            # then it is a binary column.
            if self.data[c].nunique() == 2:
                self.types[c] = Feature.BINARY
                
                # Replace the unique values with 0/1.
                values = self.data[c].unique()
                self.data[c].replace({
                    values[0]: 0,
                    values[1]: 1
                }, inplace=True)

    def __identify_numerical_features(self):
        for c in self.features:
            # Skip the column if it is already
            # identified as a feature.
            if c in self.types:
                continue
            
            # Get the column from the data.
            x = self.data[c]

            # If the column type is already
            # identified as a categorical column
            # then skip the column.
            if x.dtype == 'category':
                continue

            # Try to convert the column to a numeric
            # column. If the conversion is successful
            # then the column is a numerical column.
            # Otherwise, the column is not a numerical.
            x = pd.to_numeric(x, errors='coerce')
            if not x.notnull().all():
                continue

            # Set the column type to numerical.
            self.types[c] = Feature.NUMERICAL

            # Normalize the column to the range [0, 1].
            # This is done by taking scaling the values
            # of the column using tanh.
            dx = x.apply(ptanh).astype(float)
            self.data[c].replace(dx, inplace=True)

    def __identify_categorical_features(self):
        for c in self.features:
            # Skip the column if it is already
            # identified as a feature.
            if c in self.types:
                continue
            
            # Get the column from the data.
            x = self.data[c]
            
            # Set the column type to categorical.
            self.types[c] = Feature.CATEGORICAL
            df = pd.get_dummies(x, prefix=c)
            self.cat[c] = list(df.columns)
            
            # Drop the original column
            self.data.drop(c, axis=1, inplace=True)
            
            # Concatenate the one-hot encoded columns
            # to the data.
            self.data = pd.concat(
                [self.data, df],
                axis=1
            )