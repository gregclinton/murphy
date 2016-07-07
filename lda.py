import numpy as np
from numpy.linalg import inv

def fit(X, y):
    Sigma = np.cov(X, rowvar = False)
    InvSigma = inv(Sigma)

    def theta(c):
        prior = np.mean(y == c)
        mean = np.mean(X[y == c], axis = 0)
        gamma = -mean.dot(InvSigma).dot(mean) / 2.0 + np.log(prior)
        Beta = InvSigma.dot(mean)
        return c, gamma, Beta

    return [theta(c) for c in np.unique(y)]

def predict(model, X):
    p = []
    cs = []
        
    for theta in model:
        c, gamma, Beta = theta
        cs.append(c)
        p.append(X.dot(Beta) + gamma)
    return np.array([cs[i] for i in np.argmax(p, axis = 0)])