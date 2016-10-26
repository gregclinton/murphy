import numpy as np
from scipy.optimize import minimize
from softmax import softmax, log_softmax
from sklearn import preprocessing 
from datasets import one_hot

def crossentropy_loss(X, Y, decode):
    N, D = X.shape
    N, C = Y.shape

    def loss(params):
        W, b = decode(params)
        return -sum([Y[i].dot(ll) for i, ll in enumerate(log_softmax(X.dot(W) + b))])

    def grad(params):
        W, b = decode(params)
        return sum([np.kron(mu - Y[i], X[i]) for i, mu in enumerate(softmax(X.dot(W) + b))])

    def hess(params):
        W, b = decode(params)
        o = lambda x: np.outer(x, x)
        return sum([np.kron(np.diag(mu) - o(mu), o(X[i])) for i, mu in enumerate(softmax(X.dot(W) + b))])
    
#    return loss, grad, hess
    return loss, None, None

def hinge_loss(X, Y, decode):
    y = np.argmax(Y, axis = 1)
    rng = np.arange(len(X)) 
    
    def loss(params):
        W, b = decode(params)
        s = X.dot(W) + b
        s = (s.T - s[rng, y]).T
        s[rng, y] = 0 
        return np.sum(np.maximum(0, s + 1))
    
    return loss, None, None

class Classifier:
    def fit(self, X, y, loss):
        Y = one_hot(y)
        N, D = X.shape
        N, C = Y.shape

        # self.scaler = preprocessing.StandardScaler().fit(X)
        # X = self.scaler.transform(X)

        params = [0] * (D + 1) * C
        decode = lambda params: (params[:-C].reshape(D, C), params[-C:])

        loss, grad, hess = loss(X, Y, decode)
        
        if hess != None:
            params = minimize(loss, params, method = 'Newton-CG', jac = grad, hess = hess).x
        else:
            params = minimize(loss, params).x
        self.theta = decode(params)
    
    def predict_log_proba(self, X):
        return np.log(self.predict_proba(X))

    def predict_proba(self, X):
        # X = self.scaler.transform(X)
        W, b = self.theta
        return softmax(X.dot(W) + b)

    def predict(self, X):
        p = self.predict_proba(X)
        return np.argmax(p, axis = 1)