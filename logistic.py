import numpy as np
from scipy.optimize import minimize
from softmax import softmax, log_softmax
from sklearn import preprocessing 

def categorical_cross_entropy_loss(X, Y, penalty):
    N, D = X.shape
    N, C = Y.shape
    V0_inv = penalty * np.eye(D)

    eta = lambda W, b: X.dot(W) + np.tile(b, (N, 1))
    mus = lambda W, b: enumerate(softmax(eta(W, b)))
    decode = lambda P: (P[:-C].reshape(D, C), P[-C:])

    def loss(P):
        W, b = decode(P)
        loss = -sum([Y[i].dot(ll) for i, ll in enumerate(log_softmax(eta(W, b)))])
        return loss + (0.5 * sum([w.dot(V0_inv).dot(w) for w in W.T]) if penalty > 0 else 0.0)

    def grad(P):
        W, b = decode(P)
        grad = sum([np.kron(mu - Y[i], X[i]) for i, mu in mus(W, b)])
        return (grad + np.tile(V0_inv.dot(np.sum(W, axis = 1)), (C, 1))).ravel()

    def hess(P):
        W, b = decode(P)
        o = lambda x: np.outer(x, x)
        hess = sum([np.kron(np.diag(mu) - o(mu), o(X[i])) for i, mu in mus(W, b)])
        return hess + np.kron(np.eye(C), V0_inv)
    
    return loss, grad, hess

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
    def fit(self, X, y, penalty = 0.0):
        Y = one_hot(y)
        N, D = X.shape
        N, C = Y.shape

        # self.scaler = preprocessing.StandardScaler().fit(X)
        # X = self.scaler.transform(X)

        loss, grad, hess = categorical_cross_entropy_loss(X, Y, penalty)
        W = minimize(loss, [0] * ((D + 1) * C)).x
        # W = minimize(loss, [0] * ((D + 1) * C), method = 'Newton-CG', jac = grad, hess = hess).x
        self.theta = W[:-C].reshape(D, C), W[-C:]

    def predict_log_proba(self, X):
        return np.log(self.predict_proba(X))

    def predict_proba(self, X):
        # X = self.scaler.transform(X)
        N, D = X.shape
        W, b = self.theta
        return softmax(X.dot(W) + np.tile(b, (N, 1)))

    def predict(self, X):
        p = self.predict_proba(X)
        return np.argmax(p, axis = 1)