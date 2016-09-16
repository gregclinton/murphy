import numpy as np
from numpy.linalg import inv
from scipy import stats
import generative

'''
naive bayes classifiers (features treated as independent given theta)
murphy pp. 84 to 89
'''

class Bernoulli(generative.Classifier):
    '''
    binary features only
    or use BernoulliNB from sklearn.naive_bayes
    '''        
    def __init__(self):
        def get_theta(X, y, N, D, C):
            theta = np.empty((C, D))
            pseudocount = 1

            for c in range(C):
                X_c = X[y == c]
                Non = np.sum(X_c == 1, axis = 0)
                Noff = np.sum(X_c == 0, axis = 0)
                theta[c, :] = 1.0 * (Non + pseudocount) / (Non + Noff + 2 * pseudocount)

            return theta

        def fill_log_likelihood(X, N, D, C, theta, log_likelihood):
            for c in range(C):
                p = theta[c]
                log_likelihood[:, c] = sum([stats.bernoulli.logpmf(X[:, j], p[j]) for j in range(D)])

        generative.Classifier.__init__(self, get_theta, fill_log_likelihood)


class Gaussian(generative.Classifier):
    '''
    or use GaussianNB from sklearn.naive_bayes
    '''        
    def __init__(self):
        def get_theta(X, y, N, D, C):
            theta = []

            for c in range(C):
                X_c = X[y == c]
                mu = np.mean(X_c, axis = 0)
                sigma = np.std(X_c, axis = 0)
                theta.append(zip(mu, sigma))

            return theta

        def fill_log_likelihood(X, N, D, C, theta, log_likelihood):
            for c in range(C):
                for j in range(D):
                    mu, sigma = theta[c][j]
                    log_likelihood[:, c] += stats.norm.logpdf(X[:, j], mu, sigma)

        generative.Classifier.__init__(self, get_theta, fill_log_likelihood)