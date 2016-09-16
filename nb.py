import numpy as np
from numpy.linalg import inv
from scipy import stats
import generative

class Classifier(generative.Classifier):
    '''
    naive bayes classifiers (features treated as independent given theta)
    murphy pp. 84 to 89
    or use GaussianNB, BernoulliNB from sklearn.naive_bayes
    '''
    def __init__(self, get_theta, logp):

        def fill_log_likelihood(X, N, D, C, theta, log_likelihood):
            for c in range(C):
                log_likelihood[:, c] = sum([self.logp(X[:, j], theta[c][j]) for j in range(D)])

        self.logp = logp
        generative.Classifier.__init__(self, get_theta, fill_log_likelihood)        

def bernoulli():
    def get_theta(X, y, N, D, C):
        theta = np.empty((C, D))
        pseudocount = 1

        for c in range(C):
            X_c = X[y == c]
            Non = np.sum(X_c == 1, axis = 0)
            Noff = np.sum(X_c == 0, axis = 0)
            theta[c, :] = 1.0 * (Non + pseudocount) / (Non + Noff + 2 * pseudocount)

        return theta
        
    def logp(feature, theta):
        p = theta
        return stats.bernoulli.logpmf(feature, p)
        
    return Classifier(get_theta, logp)

def gaussian():
    def get_theta(X, y, N, D, C):
        theta = []

        for c in range(C):
            X_c = X[y == c]
            mu = np.mean(X_c, axis = 0)
            sigma = np.std(X_c, axis = 0)
            theta.append(zip(mu, sigma))

        return theta

    def logp(feature, theta):
        mu, sigma = theta
        return stats.norm.logpdf(feature, mu, sigma)

    return Classifier(get_theta, logp)