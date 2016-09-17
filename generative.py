import numpy as np

class Classifier:
    def __init__(self, get_theta, fill_log_likelihood):
        self.get_theta = get_theta
        self.fill_log_likelihood = fill_log_likelihood

    def fit(self, X, y):
        N, D = X.shape
        C = len(np.unique(y))
        self.theta = self.get_theta(X, y, N, D, C)
        self.prior = [np.mean(y == c) for c in range(C)]

    def predict_log_proba(self, X):
        N, D = X.shape
        C = len(self.prior)
        log_prior = np.log(self.prior)
        log_post = np.empty((N, C))
        self.fill_log_likelihood(X, N, D, C, self.theta, log_post)

        for c in range(C):
            log_post[:, c] += log_prior[c]

        return log_post

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))
    
    def predict(self, X):
        return np.argmax(self.predict_log_proba(X), axis = 1)