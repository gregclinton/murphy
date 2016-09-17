import numpy as np
from sigmoid import sigmoid

minimize = None

def fit(X, y):   
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
    return w0, minimize(w, NLL, g, H)

def predict(model, X):
    w0, w = model
    X.dot(w)
    return (sigmoid(w0 + X.dot(w)) > 0.5) * 1


import numpy as np

class Classifier:
    def __init__(self):
        pass

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
        self.theta = w0, minimize(w, NLL, g, H)
    
    def predict(self, X):
        w0, w = self.theta
        X.dot(w)
        return (sigmoid(w0 + X.dot(w)) > 0.5) * 1