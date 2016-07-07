import numpy as np

# np.cov(X, rowvar = False, ddof = 0)

def cov(X):
    N, D = X.shape
    N *= 1.0
    mean = np.mean(X, axis = 0).reshape((D, 1))
    return X.T.dot(X) / N - mean.dot(mean.T)

def cov(X):
    N, D = X.shape
    N *= 1.0
    mean = np.mean(X, axis = 0).reshape((D, 1))
    Sigma = np.zeros((D, D))
    
    for x in X:
        Sigma += (x.reshape((D, 1)) - mean) ** 2
        
    return Sigma / N