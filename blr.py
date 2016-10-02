import numpy as np
from scipy.optimize import minimize
from sigmoid import sigmoid
from scipy import stats

class Classifier:
    '''
    bayesian logistic regression classifier
    murphy p. 255
    '''
    def __init__(self):
        pass
        
    def preprocess(self, X):
        X = X.astype(float)
        return (X - self.mean) / np.sqrt(np.diag(self.cov))
        
    def fit(self, X, y):
        N, D = X.shape
        C = len(np.unique(y))

        X = X.astype(float)
        self.mean = np.mean(X, axis = 0)
        self.cov = np.cov(X, ddof = 0, rowvar = False)
        
        X = self.preprocess(X)
        
        def ll(w):
            muw = mu(w)
            return sum(y * np.log(muw) + (1 - y) * np.log(1 - muw))

        ybar = np.mean(y)
        w0 = np.log(ybar / (1 - ybar))
        mu = lambda w: sigmoid(w0 + X.dot(w))
        mvn = stats.multivariate_normal
        log_prior = lambda w: mvn.logpdf(w, np.zeros(D), np.eye(D) * 100)
        f = lambda w: -ll(w) - log_prior(w) 

        w = minimize(f, [0] * D, bounds = [(-5, 5), (-5, 5)]).x
        self.theta = w0, w
            
    def predict_log_proba(self, X):
        return np.log(self.predict_proba(X))

    def predict_proba(self, X):
        X = self.preprocess(X)
        w0, w = self.theta
        return sigmoid(w0 + X.dot(w))

    def predict(self, X):
        return (self.predict_proba(X) > 0.5) * 1