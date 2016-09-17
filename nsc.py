import numpy as np
import math
import generative

def soft_threshold(a, delta):
    return np.array([np.sign(x) * max(np.abs(x) - delta, 0) for x in a])

class Classifier(generative.Classifier):
    '''
    nearest shrunken centroids classifier
    murphy p. 109
    '''        
    def __init__(self, delta = 3.0):
        self.delta = delta
        
        def get_theta(X, y, N, D, C):
            sse = np.zeros(D)
            mus = []
            xbar = np.mean(X, axis = 0)

            for c in range(C):
                X_c = X[y == c]
                mu_c = np.mean(X_c, axis = 0)
                mus.append(mu_c)
                sse += np.sum((X_c - mu_c) ** 2, axis = 0)

            sigma2 = sse / (1.0 * N - C)
            sigma = np.sqrt(sigma2)
            s0 = np.median(sigma)

            for c in range(C):
                m = math.sqrt(1.0 / sum(y == c) - 1.0 / N)
                d = (mus[c] - xbar) / (m * (sigma + s0))
                d = soft_threshold(d, self.delta)
                mus[c] = xbar + m * (sigma + s0) * d

            return mus, sigma2

        def fill_log_likelihood(X, N, D, C, theta, log_likelihood):
            mus, sigma2 = theta

            for c in range(C):
                Z = 0.5 * (X - mus[c]) ** 2 / sigma2
                log_likelihood[:, c] = - np.sum([Z[:, j] for j in range(D)], axis = 0)
        
        generative.Classifier.__init__(self, get_theta, fill_log_likelihood)