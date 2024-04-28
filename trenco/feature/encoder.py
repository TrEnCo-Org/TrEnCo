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

    # Feature names before encoding.
    features: pd.Index[str]

    # Column names after encoding.
    columns: pd.Index[str]

    # Categories of the categorical columns.
    cat: dict[str, list[str]]

    # Inverse mapping of the categorical columns.
    # This is used to map the one-hot encoded columns
    # to the original categorical columns.
    inv_cat: dict[str, str]

    # Encoded data.
    X: pd.DataFrame
    
    def ___init__(self):
        pass
    
    def fit(self, data: pd.DataFrame):
        self.data = deepcopy(data)
        
        # clean the data
        self.clean()

        # Get the columns of the data.
        self.features = self.data.columns

        # Initialize the features dictionary.
        self.types = dict()
        self.cat = dict()
        self.inv_cat = dict()
        
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
        self.fit_bin_features()
        self.fit_num_features()
        self.fit_cat_features()

        # Set the data after encoding.
        self.X = self.data
        
        # Save the column names after encoding.
        self.columns = self.X.columns
        
    def transform(self):
        return self.X

    def clean(self):
        # Remove rows with missing values
        self.data.dropna(inplace=True)
        
        # Remove columns with only one unique value
        b = self.data.nunique() > 1
        self.data = self.data.loc[:, b]
        
    def fit_bin_features(self):
        for f in self.features:
            # If the column has only two unique values
            # then it is a binary column.
            if self.data[f].nunique() == 2:
                self.types[f] = Feature.BINARY
                
                # Replace the unique values with 0/1.
                values = self.data[f].unique()
                self.data[f].replace({
                    values[0]: 0,
                    values[1]: 1
                }, inplace=True)

    def fit_num_features(self):
        for f in self.features:
            # Skip the column if it is already
            # identified as a feature.
            if f in self.types:
                continue
            
            # Get the column from the data.
            x = self.data[f]

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
            self.types[f] = Feature.NUMERICAL

            # Normalize the column to the range [0, 1].
            # This is done by taking scaling the values
            # of the column using positive tanh.
            p = x.apply(ptanh).astype(float)
            self.data[f].replace(p, inplace=True)

    def fit_cat_features(self):
        for f in self.features:
            # Skip the column if it is already
            # identified as a feature.
            if f in self.types:
                continue
            
            # Get the column from the data.
            x = self.data[f]
            
            # Set the column type to categorical.
            self.types[f] = Feature.CATEGORICAL
            df = pd.get_dummies(x, prefix=f)
            self.cat[f] = list(df.columns)
            for c in df.columns:
                self.inv_cat[c] = f
            
            # Drop the original column
            self.data.drop(f, axis=1, inplace=True)
            
            # Concatenate the one-hot encoded columns
            # to the data.
            self.data = pd.concat([self.data, df], axis=1)