import numpy as np
from numpy.linalg import inv
from sigmoid import sigmoid

class Classifier:
    '''
    iteratively reweighted least squares
    murphy p. 253
    '''        
    def fit(self, X, y):
        N, D = X.shape
        w = np.zeros(D)
        ybar = np.mean(y)
        w0 = np.log(ybar / (1 - ybar))

        for k in range(20):
            eta = w0 + X.dot(w)
            mu = sigmoid(eta)
            s = mu * (1 - mu)
            z = eta + (y - mu) / s
            S = np.diag(s)
            w = inv(X.T.dot(S).dot(X)).dot(X.T).dot(S).dot(z)    
        self.theta = w0, w

    def predict_log_proba(self, X):
        return np.log(self.predict_proba(X))

    def predict_proba(self, X):
        w0, w = self.theta
        X.dot(w)
        return sigmoid(w0 + X.dot(w))

    def predict(self, X):
        return (self.predict_proba(X) > 0.5) * 1