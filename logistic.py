import numpy as np
from scipy import stats
import newton
from sigmoid import sigmoid

class Classifier:
    def __init__(self):
        self.minimize = newton.minimize

    def fit(self, X, y):
        N, D = X.shape
        penalty = 0.1

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
        self.theta = w0, self.minimize(w, NLL, g, H)

    def predict_log_proba(self, X):
        return np.log(self.predict_proba(X))
        
    def predict_proba(self, X):
        w0, w = self.theta
        X.dot(w)
        return sigmoid(w0 + X.dot(w)
    
    def predict(self, X):
        return (self.predict_proba(X) > 0.5) * 1