# TrEnCo

TrEnCo is a tool for pruning and compressing Tree Ensembles (Random Forests, Gradient Boosted Trees, etc.).

## Installation

To install and use TrEnCo, you need to clone this repository:

```bash
git clone https://www.github.com/TrEnCo-Org/TrEnCo.git
```

Then, you need to install the required packages:

```bash
pip install -e .
```

## Usage

Currently TrEnCo supports pruning and compression of tree ensembles that are a `Iterable` of `DecisionTreeClassifier` from `scikit-learn`.

First you need to load your dataset using the `DatasetLoader` class. This class takes as input the path to an csv file that contains the data, the path to a csv file that contains the feature types and the name of the class column. 

```python
from trenco.utils.data import DatasetLoader

dataset = DatasetLoader(
    'path/to/dataset.full.csv',
    'path/to/dataset.types.csv',
    label='Class'
)
```

You can split the dataset into training and testing sets using the `split` method of the `DatasetLoader` class. This method takes as input the ratio of the training set and the fraction of the dataset to be used for both the training and testing sets.

```python
# For example: 
# to split half of the dataset
# into training and testing sets.
# with 80% used for training
# and 20% for testing.
train, test = dataset.split(
    ratio=0.8,
    frac=0.5
)
```

You can use the `train` and `test` datasets to train and evaluate your model. For example to train a Random Forest model, you can use the `train` set as follows:

```python
from sklearn.ensemble import RandomForestClassifier
X, y = train.values
model = RandomForestClassifier()
model.fit(X, y)
```

Before pruning the model, you need to wrap the model in a `Classifier` object. This object takes as input the model.

```python
from trenco.model.ensemble import Classifier
clf = Classifier(model)
```

By default the tie-breaking rule is to take the first class in case of a tie. You can change this behavior by passing the `comp` argument to the `Classifier` object. For example, to take the last class in case of a tie, you can do the following:

```python
from trenco.typing import Comparator
from trenco.model.ensemble import Classifier

comp = Comparator(lambda x, y: x > y)
clf = Classifier(model, comp=comp)
```

Finally you can prune the model using the `PrunerMIP` class. This class takes as input the `Classifier` object and the `train` set. For example, to prune the model on train dataset, you can do the following:

```python
from trenco.pruning import PrunerMIP

pruner = PrunerMIP(clf, train)
pruner.build()
pruner.prune()

# You can get the pruned model 
# using the `pruned` attribute.
pruned_model = pruner.pruned
```

You can also prune the model by passing the validation set to the `PrunerMIP` class. For example, to prune the model on the validation dataset, you can do the following:

```python
from trenco.pruning import PrunerMIP

train, valid, test = dataset.split(
    ratio=[0.7, 0.15],
    frac=0.5
)

pruner = PrunerMIP(clf, train, valid)
pruner.build()
pruner.prune()
```

This will prune the model initially on the training set and then on consecutively on the validation set.