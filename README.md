# TrEnCo

TrEnCo is a tool for pruning and compressing Ensembles (Random Forests, Gradient Boosted Trees, etc.).

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

Currently TrEnCo supports pruning and compression of any ensembles that are a `RandomForestClassifier` or `VotingClassifier` from `scikit-learn`.

You can refer to the `examples` folder for examples on how to use TrEnCo.