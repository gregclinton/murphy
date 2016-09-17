import numpy as np
from scipy import stats
from scipy.optimize import minimize
from sigmoid import sigmoid

class Classifier:
    '''
    logistic regression classifier
    murphy p. 255
    '''
    def __init__(self, C = 1):
        self.C = C
        
    def fit(self, X, y):
        N, D = X.shape
        penalty = 1.0 / self.C

        def NLL(w):
            muw = mu(w)
            return -sum(y * np.log(muw) + (1 - y) * np.log(1 - muw)) + penalty * w.dot(w)

        ybar = np.mean(y)
        w0 = np.log(ybar / (1 - ybar))
        mu = lambda w: sigmoid(w0 + X.dot(w))
        g = lambda w: X.T.dot(mu(w) - y) + 2 * penalty * w
        S = lambda w, mu: np.diag(mu * (1 - mu))
        H = lambda w: X.T.dot(S(w, mu(w))).dot(X) + 2 * penalty * np.eye(D)
        w = np.zeros(D)
        self.theta = w0, minimize(NLL, w, method = 'Newton-CG', jac = g, hess = H).x

    def predict_log_proba(self, X):
        return np.log(self.predict_proba(X))

    def predict_proba(self, X):
        w0, w = self.theta
        X.dot(w)
        return sigmoid(w0 + X.dot(w))

    def predict(self, X):
        return (self.predict_proba(X) > 0.5) * 1