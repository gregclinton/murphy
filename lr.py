import numpy as np
import math
from numpy.linalg import inv
from sigmoid import sigmoid

def fit(X, y):
    heads = y == 0
    prior_0 = np.mean(heads)
    prior_1 = 1.0 - prior_0
    X_0 = X[heads]
    X_1 = X[np.logical_not(heads)]
    mu_0 = np.mean(X_0, axis = 0)
    mu_1 = np.mean(X_1, axis = 0)

    a = mu_1 + mu_0
    b = mu_1 - mu_0
    InvSigma = inv(np.cov(X, rowvar = False))
    w = InvSigma.dot(b)
    x_0 = 0.5 * a - b * math.log(prior_1 / prior_0) / b.T.dot(InvSigma).dot(b)
    return w, x_0

def predict(model, X):
    w, x_0 = model
    return 1 * (sigmoid(w.dot((X - x_0).T)) > 0.5)