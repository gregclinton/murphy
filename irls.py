import numpy as np
from numpy.linalg import inv
from sigmoid import sigmoid

def fit(X, y):
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
    return w0, w

def predict(model, X):
    w0, w = model
    return (sigmoid(w0 + X.dot(w)) > 0.5) * 1