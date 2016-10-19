import numpy as np
from scipy.optimize import minimize
from softmax import softmax, log_softmax
from sklearn import preprocessing 

def categorical_cross_entropy_loss(X, Y, decode, eta, penalty):
    N, D = X.shape
    N, C = Y.shape
    V0_inv = penalty * np.eye(D)

    mus = lambda W, b: enumerate(softmax(eta(X, W, b)))

    def loss(params):
        W, b = decode(params)
        loss = -sum([Y[i].dot(ll) for i, ll in enumerate(log_softmax(eta(X, W, b)))])
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

def categorical_svm_loss(X, Y, decode, eta, penalty):
    N, D = X.shape
    N, C = Y.shape
    
    penalty = 0.1

    def loss(params):
        W, b = decode(params)
        s = eta(X, W, b)
        
        def L(i):
            c = np.argmax(Y[i])
            margins = np.maximum(0, s[i] - s[i, c] + 1)
            margins[c] = 0
            return sum(margins)
        
        return sum([L(i) for i in range(N)]) + np.sum(W ** 2) * penalty
    
    return loss, None, None

def one_hot(y):
    y = np.array(y)
    if y.ndim == 2:
        return y
    else:
        N, C = len(y), len(np.unique(y))
        Y = np.zeros((N, C))
        for i in range(N):
            Y[i, int(y[i])] = 1
        return Y

class Classifier:
    def fit(self, X, y, loss, penalty = 0.0):
        Y = one_hot(y)
        N, D = X.shape
        N, C = Y.shape

        # self.scaler = preprocessing.StandardScaler().fit(X)
        # X = self.scaler.transform(X)

        params = [0] * (D + 1) * C
        decode = lambda params: (params[:-C].reshape(D, C), params[-C:])

        loss, grad, hess = loss(X, Y, decode, self.eta, penalty)
        
        if hess != None:
            params = minimize(loss, params, method = 'Newton-CG', jac = grad, hess = hess).x
        else:
            params = minimize(loss, params).x
        self.theta = decode(params)

    @staticmethod
    def eta(X, W, b):
        return X.dot(W) + np.tile(b, (len(X), 1))
    
    def predict_log_proba(self, X):
        return np.log(self.predict_proba(X))

    def predict_proba(self, X):
        # X = self.scaler.transform(X)
        W, b = self.theta
        return softmax(self.eta(X, W, b))

    def predict(self, X):
        p = self.predict_proba(X)
        return np.argmax(p, axis = 1)