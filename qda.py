import numpy as np
from scipy import stats

def fit(X, y):
    def theta(c):
        X_c = X[y == c]
        prior = 1.0 * sum(y == c) / len(y)
        mean = np.mean(X_c, axis = 0)
        Sigma = np.cov(X_c, rowvar = False)
        return c, prior, mean, Sigma

    return [theta(c) for c in np.unique(y)]

def predict(model, X):
    p = []
    cs = []
        
    for theta in model:
        c, prior, mean, Sigma = theta
        cs.append(c)
        p.append(prior * stats.multivariate_normal.pdf(X, mean, Sigma))
            
    return np.array([cs[i] for i in np.argmax(p, axis = 0)])
    
# from sklearn import qda
# clf = qda.QDA()