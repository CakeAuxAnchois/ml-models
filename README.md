# ML-models

This provides an easily readable python implementation of machine learning
models.

## Installation

```
$ python3 setup.py install
```

## Support Vector Machine

Binary SVM handles rbf and polynomial kernels and is solved by Quadratic Programming
using CVXOPT.
The multiclass SVM is based on One-vs-One model.

The model implemented here performs as well as the scikit-learn one:

```
$ python3 tests/svm_vs_sckikit.py
$ Model accuracy : 0.986072423398
$ scikit-learn model accuracy : 0.986072423398
```
