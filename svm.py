import numpy as np
import cvxopt

class BinarySVM:

    def __init__(self, kernel, param, C=None):
        self._param = param
        self._C = C
        self._lagrange = None
        self._labels = None
        self._b = None

        if kernel == 'rbf':
            self._kernel = self.RBF
        elif kernel == 'poly':
            self._kernel = self.poly
        else:
            raise ValueError('Unknown kernel')

    def fit(self, x, labels):
        gram_matrix = self._kernel(x, x, self._param)
        print("gram: \n", gram_matrix[0:2, 0:2])
        print(x[0:2, :])
        print(gram_matrix.shape)
        self._labels = labels

        n_samples, n_features = x.shape
        P = cvxopt.matrix(np.outer(labels, labels) * gram_matrix)
        q = cvxopt.matrix(- np.ones(n_samples))
        G = np.eye(n_samples) * -1
        h = np.zeros(n_samples)

        A = cvxopt.matrix(labels, tc='d').T
        b = cvxopt.matrix(0.0)

        if self._C is not None:
            G_max = np.eye(n_samples)
            h_max = np.ones(n_samples) * self._C
            G = cvxopt.matrix(np.concatenate((G, G_max), axis=0))
            h = cvxopt.matrix(np.concatenate((h, h_max), axis=0))

        G = cvxopt.matrix(G)
        h = cvxopt.matrix(h)

        cvxopt.solvers.options['abstol']= 1e-20

        self._lagrange = cvxopt.solvers.qp(P, q, G, h, A, b)
        self._lagrange = np.array(self._lagrange['x']).reshape(n_samples)

        print("lagrange", np.max(self._lagrange))

        epsilon = np.max(self._lagrange / 100)
        idx_support = self._lagrange > epsilon

        self._support_v = x[idx_support]
        self._lagrange = self._lagrange[idx_support]
        self._labels = self._labels[idx_support]

        n_support_v = len(self._support_v)
        gram_matrix = gram_matrix[idx_support]
        gram_matrix = gram_matrix[:, idx_support]

        self._b = (self._labels[0]
                   - np.sum((self._labels * self._lagrange).T * gram_matrix[0]))

    def predict(self, x):
        if self._lagrange is None:
            raise ValueError('Model not trained')
        gram_matrix = self._kernel(x, self._support_v, self._param)
        prediction = self._lagrange * self._labels * gram_matrix
        prediction = np.sum(prediction, axis=1)+ self._b

        prediction = np.sign(prediction)

        return prediction

    @staticmethod
    def RBF(x, y, gamma):
        gram_matrix = -2 * np.dot(x, y.T)
        norm_sqrd_x = np.sum(x ** 2, axis=1)
        norm_sqrd_y = np.sum(y ** 2, axis=1)
        gram_matrix += norm_sqrd_x.reshape(-1, 1) + norm_sqrd_y.reshape(1, -1)

        gram_matrix = np.exp(-gamma * gram_matrix)

        return gram_matrix

    @staticmethod
    def poly(x, y, d):
        return (np.dot(x, y.T) + 1) ** d
