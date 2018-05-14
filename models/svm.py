import numpy as np
import cvxopt

class BinarySVM:

    def __init__(self, kernel, C=None, degree=3, gamma=0.001):
        self._degree = degree
        self._gamma = gamma
        self._C = C
        self._lagrange = None
        self._labels = None
        self._labels_list = None
        self._b = None

        if kernel == 'rbf':
            self._kernel = self.RBF(gamma)
        elif kernel == 'poly':
            self._kernel = self.poly(degree)
        else:
            raise ValueError('Unknown kernel')

    def fit(self, x, labels):
        self._labels_list = np.unique(labels)
        self._labels = labels

        if len(self._labels_list) != 2:
            raise ValueError('Binary SVM must receive exactly 2 classes')

        idx_c1 = labels == self._labels_list[0]
        idx_c2 = labels == self._labels_list[1]

        self._labels[idx_c1] = 1
        self._labels[idx_c2] = -1

        gram_matrix = self._kernel(x, x)

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

        cvxopt.solvers.options['abstol']= 1e-25
        cvxopt.solvers.options['show_progress']= False

        self._lagrange = cvxopt.solvers.qp(P, q, G, h, A, b)
        self._lagrange = np.array(self._lagrange['x']).reshape(n_samples)

        epsilon = np.max(self._lagrange / 100)
        idx_support = self._lagrange > epsilon

        self._support_v = x[idx_support]
        self._lagrange = self._lagrange[idx_support]
        self._labels = self._labels[idx_support]

        n_support_v = len(self._support_v)
        gram_matrix = gram_matrix[idx_support]
        gram_matrix = gram_matrix[:, idx_support]

        self._b = np.sum(self._labels
                         - (np.dot((self._labels * self._lagrange).T,
                                   gram_matrix)))
        self._b /= n_support_v

    def predict(self, x):
        if self._lagrange is None:
            raise ValueError('Model not trained')

        gram_matrix = self._kernel(x, self._support_v)
        prediction = self._lagrange * self._labels * gram_matrix
        prediction = np.sum(prediction, axis=1) + self._b

        prediction = np.sign(prediction)

        idx_c1 = prediction == 1
        idx_c2 = prediction == -1

        prediction[idx_c1] = self._labels_list[0]
        prediction[idx_c2] = self._labels_list[1]
        return prediction

    def score(self, x, labels):
        prediction = self.predict(x)

        return np.sum(prediction == labels) / len(labels)

    # Matrix implementations of kernel functions

    @staticmethod
    def RBF(gamma):
        def f(x, y):
            gram_matrix = -2 * np.dot(x, y.T)
            norm_sqrd_x = np.sum(x ** 2, axis=1)
            norm_sqrd_y = np.sum(y ** 2, axis=1)
            gram_matrix += norm_sqrd_x.reshape(-1, 1) + norm_sqrd_y.reshape(1, -1)

            gram_matrix = np.exp(-gamma * gram_matrix)

            return gram_matrix
        return f

    @staticmethod
    def poly(degree):
        def f(x, y):
            return (np.dot(x, y.T) + 1) ** d
        return f
