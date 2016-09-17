import numpy as np
from scipy import stats
import generative

mvn = stats.multivariate_normal

class Classifier(generative.Classifier):
    '''
    quadratic discriminant analysis classifier
    murphy p. 102
    or use QDA from sklearn.qda
    '''        
    def __init__(self):
        def get_theta(X, y, N, D, C):
            means = []
            Sigmas = []

            for c in range(C):
                X_c = X[y == c]
                means.append(np.mean(X_c, axis = 0))
                Sigmas.append(np.cov(X_c, rowvar = False))

            return means, Sigmas

        def fill_log_likelihood(X, N, D, C, theta, log_likelihood):
            means, Sigmas = theta

            for c in range(C):
                log_likelihood[:, c] = mvn.logpdf(X, means[c], Sigmas[c])
        
        generative.Classifier.__init__(self, get_theta, fill_log_likelihood)