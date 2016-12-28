import numpy as np
from scipy import stats

mvn = stats.multivariate_normal
inv = np.linalg.inv

class SquaredExponential(object):
    def __init__(self, tau, l):
        self.tau = tau
        self.l = l

    def __call__(self, x, x_prime):
        return self.tau ** 2 * np.exp(-(np.linalg.norm(x - x_prime) ** 2 / self.l ** 2))

def zero(x):
    return np.zeros(len(x))

class K(object):
    def __init__(self, kernel):
        self.k = kernel

    def __call__(self, x, x_prime):
        return np.reshape([self.k(p, q) for p in x for q in x_prime], (len(x), len(x_prime)))
    
def mu(x, y, x_prime, theta, kernel):
    KK = K(kernel)
    xpx = KK(x_prime, x)
    M = xpx.dot(inv(KK(x, x) + theta ** 2 * np.identity(len(x))))
    mean = M.dot(y)
    C = KK(x_prime, x_prime) - M.dot(xpx.transpose()) 
    return mvn(mean, C, allow_singular = True)