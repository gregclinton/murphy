import numpy as np
from numpy.linalg import inv
from sigmoid import sigmoid
from scipy.optimize import minimize

def fit(X, y):
    N, D = X.shape
    w = np.zeros(D)
    ybar = np.mean(y)
    w0 = np.log(ybar / (1 - ybar))

    for k in range(2):
        eta = w0 + X.dot(w)
        mu = sigmoid(eta)
        s = mu * (1 - mu)
        z = eta + (y - mu) / s
        S = np.diag(s)
        w = inv(X.T.dot(S).dot(X)).dot(X.T).dot(z)    
    return w0, w

def predict(model, X):
    w0, w = model
    return (sigmoid(w0 + X.dot(w)) > 0.5) * 1


def fit(X, y):
    N, D = X.shape
    
    def NLL(w):
        muw = mu(w)
        return -sum(y * np.log(muw) + (1 - y) * np.log(1 - muw))

    mu = lambda w: sigmoid(X.dot(w))
    jac = lambda w: X.T.dot(mu(w) - y)
    S = lambda w, mu: np.diag(mu * (1 - mu))
    hess = lambda w: X.T.dot(S(w, mu(w))).dot(X)

    w = np.ones(D)
    return minimize(NLL, w, jac = jac, hess = hess, method = 'Newton-CG').x

def predict(model, X):
    w = model
    X.dot(w)
    return (sigmoid(X.dot(w)) > 0.5) * 1