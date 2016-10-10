import numpy as np
from numpy.linalg import inv
from scipy.special import expit

class Classifier:
    '''
    Fisher's linear discriminant analysis
    murphy p. 271
    '''
    def preprocess(self, X):
        return X
        # see sklearn.preprocessing.scale
        X = X.astype(float)
        return (X - self.mean) / np.sqrt(np.diag(self.cov))
        
    def fit(self, X, y):
        N, D = X.shape
        C = len(np.unique(y))
        mus = []

        ybar = np.mean(y)
        w0 = np.log(ybar / (1.0 - ybar))
        
        X = X.astype(float)
        self.mean = np.mean(X, axis = 0)
        self.cov = np.cov(X, ddof = 0, rowvar = False)
        
        X = self.preprocess(X)
        
        SW = np.zeros((D, D))
        aaT = lambda a: np.outer(a, a)
        
        for c in range(C):
            X_c = X[y == c]
            mu = np.mean(X_c, axis = 0)
            for x in X_c:
                SW += aaT(x - mu)
            mus.append(mu)
            
        w = inv(SW).dot(mus[1] - mus[0])
        self.theta = w0, w
            
    def predict_log_proba(self, X):
        return np.log(self.predict_proba(X))

    def predict_proba(self, X):
        X = self.preprocess(X)
        w0, w = self.theta
        return expit(w0 + X.dot(w))

    def predict(self, X):
        p = self.predict_proba(X)
        return (p > 0.5) * 1