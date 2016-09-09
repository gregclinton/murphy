import numpy as np
from sigmoid import sigmoid
from scipy.optimize import minimize
from numpy.linalg import inv, norm

class Converge:
    def __init__(self, f, theta, epsilon):
        self.f = f
        self.epsilon = epsilon
        self.mark = f(theta)
                
    def done(self, theta):
        mark = self.f(theta)
        res = abs(mark - self.mark) < self.epsilon
        self.mark = mark
        return res
    
def newton(theta, f, g, H, epsilon = 0.0000001):
    converge = Converge(f, theta, epsilon)
    
    for _ in range(100):
        gk = g(theta)
        eta = minimize(lambda eta: f(theta - eta * gk), 1).x
        theta -= eta * inv(H(theta)).dot(gk)
        if converge.done(theta):
            break
    return theta

def fit(X, y):   
    def NLL(w):
        muw = mu(w)
        return -sum(y * np.log(muw) + (1 - y) * np.log(1 - muw))

    ybar = np.mean(y)
    w0 = np.log(ybar / (1 - ybar))
    mu = lambda w: sigmoid(w0 + X.dot(w))
    g = lambda w: X.T.dot(mu(w) - y)
    S = lambda w, mu: np.diag(mu * (1 - mu))
    H = lambda w: X.T.dot(S(w, mu(w))).dot(X)
    N, D = X.shape
    w = np.zeros(D)
    return w0, newton(w, NLL, g, H)

def predict(model, X):
    w0, w = model
    X.dot(w)
    return (sigmoid(w0 + X.dot(w)) > 0.5) * 1