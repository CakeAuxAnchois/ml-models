from models.svm import BinarySVM
import numpy as np

class MulticlassSVM():
    def __init__(self, kernel, degree=3, gamma=0.001, C=None):
        self._kernel = kernel
        self._degree = degree
        self._gamma = gamma
        self._C = C
        self._svm = None

    def fit(self, x, labels):
        self._labels_list = np.unique(labels)
        self._n_labels = len(self._labels_list)
        n_samples, n_features = x.shape

        n_svm = int(self._n_labels * (self._n_labels - 1) / 2)

        self._svm = np.array([BinarySVM(self._kernel,
                                        degree=self._degree,
                                        gamma=self._gamma,
                                        C=self._C) for _ in range(n_svm)])

        count = 0
        for i in range(self._n_labels):
            for j in range(i + 1, self._n_labels):
                class_i = self._labels_list[i]
                class_j = self._labels_list[j]
                rows_to_keep = (labels == class_i) | (labels == class_j)
                binary_labels = labels[rows_to_keep]

                svm_samples = x[rows_to_keep]
                self._svm[count].fit(svm_samples, binary_labels)
                count += 1

    def predict(self, x, test_labels):
        n_samples, _ = x.shape
        prediction = np.zeros((n_samples, self._n_labels))

        count = 0
        for i in range(self._n_labels):
            for j in range(i + 1, self._n_labels):
                class_i = self._labels_list[i]
                class_j = self._labels_list[j]

                bin_predict = self._svm[count].predict(x)

                idx = bin_predict == self._labels_list[i]
                prediction[idx, i] += 1
                idx = bin_predict == self._labels_list[j]
                prediction[idx, j] += 1

                count += 1

        prediction = np.argmax(prediction, axis=1)
        return prediction

    def score(self, x, test_labels):
        out = self.predict(x, test_labels)
        accuracy = np.sum(out == test_labels) / len(test_labels)

        return accuracy
