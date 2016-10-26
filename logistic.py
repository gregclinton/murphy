import numpy as np
from scipy.optimize import minimize
from softmax import softmax, log_softmax
from sklearn import preprocessing 
from datasets import one_hot

def categorical_crossentropy_loss(X, Y, decode, penalty):
    N, D = X.shape
    N, C = Y.shape
    V0_inv = penalty * np.eye(D)

    mus = lambda W, b: enumerate(softmax(X.dot(W) + b))

    def loss(params):
        W, b = decode(params)
        loss = -sum([Y[i].dot(ll) for i, ll in enumerate(log_softmax(X.dot(W) + b))])
        return loss + (0.5 * sum([w.dot(V0_inv).dot(w) for w in W.T]) if penalty > 0 else 0.0)

    def grad(params):
        W, b = decode(params)
        grad = sum([np.kron(mu - Y[i], X[i]) for i, mu in mus(W, b)])
        return (grad + np.tile(V0_inv.dot(np.sum(W, axis = 1)), (C, 1))).ravel()

    def hess(params):
        W, b = decode(params)
        o = lambda x: np.outer(x, x)
        hess = sum([np.kron(np.diag(mu) - o(mu), o(X[i])) for i, mu in mus(W, b)])
        return hess + np.kron(np.eye(C), V0_inv)
    
    # return loss, grad, hess
    return loss, None, None

def categorical_hinge_loss(X, Y, decode, penalty):
    y = np.argmax(Y, axis = 1)
    rng = np.arange(len(X)) 
    
    def loss(params):
        W, b = decode(params)
        s = X.dot(W) + b
        s = (s.T - s[rng, y]).T
        s[rng, y] = 0 
        return np.sum(np.maximum(0, s + 1)) # + np.sum(W ** 2) * penalty
    
    return loss, None, None

class Classifier:
    def fit(self, X, y, loss, penalty = 0.0):
        Y = one_hot(y)
        N, D = X.shape
        N, C = Y.shape

        # self.scaler = preprocessing.StandardScaler().fit(X)
        # X = self.scaler.transform(X)

        params = [0] * (D + 1) * C
        decode = lambda params: (params[:-C].reshape(D, C), params[-C:])

        loss, grad, hess = loss(X, Y, decode, penalty)
        
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