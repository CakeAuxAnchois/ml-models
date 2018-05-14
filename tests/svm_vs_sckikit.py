from sklearn.datasets import load_digits
import numpy as np
import sklearn.svm
import math
from models.multiclass_svm import MulticlassSVM
from models import utils

def main():
    data, labels = load_digits(10, True)
    n_samples, _ = data.shape

    data, labels = utils.shuffle(data, labels)
    test_idx = math.ceil(n_samples * 0.80)

    train_data = data[: test_idx]
    train_labels = labels[: test_idx]

    test_data = data[test_idx:]
    test_labels = labels[test_idx:]


    skmodel = sklearn.svm.SVC(C=15,kernel='rbf',
                              degree=3,
                              gamma=0.001,
                              shrinking=False,
                              tol=1e-12,
                              coef0=1)
    skmodel.fit(train_data, train_labels)

    model = MulticlassSVM('rbf', gamma=0.001, C=15)
    model.fit(train_data, train_labels)

    print("Model accuracy :", model.score(test_data, test_labels))
    print("Scikit-learn model accuracy :", skmodel.score(test_data, test_labels))

if __name__ == '__main__':
    main()
