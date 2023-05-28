import numpy as np


class AdalineGD:
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.1, size=X.shape[1])
        self.b_ = np.float_(0.0)
        self.losses_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = y - output

            """
                for w_j in range(self.w_.shape[0]):
                    self.w_[w_j] += self.eta * (2.0 * (X[:, w_j]*errors)).mean()
            """

            """ maps to: w := w + (-eta * dL/dWj)"""
            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            """ maps to: b := b + (-eta * dL/db)"""
            self.b_ += self.eta * 2.0 * errors.mean()
            loss = (errors**2).mean()
            self.losses_.append(loss)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_) + self.b_

    def activation(self, X):
        """Linear activation function"""
        return X

    def predict(self, X):
        """1 or 0 with a threshold of 0.5"""
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)
