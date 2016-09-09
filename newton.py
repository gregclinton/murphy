import numpy as np
from sigmoid import sigmoid
from scipy.optimize import minimize

def fit(X, y):
    N, D = X.shape
    
    def NLL(w):
        muw = mu(w)
        return -sum(y * np.log(muw) + (1 - y) * np.log(1 - muw))

    ybar = np.mean(y)
    w0 = np.log(ybar / (1 - ybar))
    
    mu = lambda w: sigmoid(w0 + X.dot(w))
    jac = lambda w: X.T.dot(mu(w) - y)
    S = lambda w, mu: np.diag(mu * (1 - mu))
    hess = lambda w: X.T.dot(S(w, mu(w))).dot(X)

    w = np.zeros(D)
    return w0, minimize(NLL, w, jac = jac, hess = hess, method = 'Newton-CG').x

def predict(model, X):
    w0, w = model
    X.dot(w)
    return (sigmoid(w0 + X.dot(w)) > 0.5) * 1