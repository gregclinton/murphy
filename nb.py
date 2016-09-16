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
    or use from sklearn.naive_bayes import BernoulliNB
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

        def get_log_likelihood(X, N, D, C, theta):
            log_likelihood = np.empty((N, C))
            
            for c in range(C):
                log_likelihood[:, c] = sum([stats.bernoulli.logpmf(X[:, j], theta[c, j]) for j in range(D)])

            return log_likelihood
                
        generative.Classifier.__init__(self, get_theta, get_log_likelihood)


class Gaussian(generative.Classifier):
    '''
    or use from sklearn.naive_bayes import GaussianNB
    '''        
    def __init__(self):
        def get_theta(X, y, N, D, C):
            mu = []
            sigma = []

            for c in range(C):
                X_c = X[y == c]
                mu.append(np.mean(X_c, axis = 0))
                sigma.append(np.std(X_c, axis = 0))
        
            return mu, sigma

        def get_log_likelihood(X, N, D, C, theta):
            mu, sigma = theta
            log_likelihood = np.empty((N, C))

            for c in range(C):
                log_likelihood[:, c] = stats.multivariate_normal.logpdf(X, mu[c], sigma[c])
            return log_likelihood
        
        generative.Classifier.__init__(self, get_theta, get_log_likelihood)        