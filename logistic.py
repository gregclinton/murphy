import numpy as np
from scipy.optimize import minimize
from softmax import softmax, log_softmax
from sklearn import preprocessing 

def categorical_cross_entropy_loss(X, Y, penalty):
    N, D = X.shape
    N, C = Y.shape

    eta = lambda W, b: X.dot(W) + np.tile(b, (N, 1))
    mus = lambda W, b: enumerate(softmax(eta(W, b)))
    decode = lambda P: (P[:-C].reshape(D, C), P[-C:])
    V0_inv = penalty * np.eye(D)

    def objective(P):
        W, b = decode(P)
        nll = -sum([Y[i].dot(ll) for i, ll in enumerate(log_softmax(eta(W, b)))])
        return nll + (0.5 * sum([w.dot(V0_inv).dot(w) for w in W.T]) if penalty > 0 else 0.0)

    def gradient(P):
        W, b = decode(P)
        grad = sum([np.kron(mu - Y[i], X[i]) for i, mu in mus(W, b)])
        return grad + np.tile(V0_inv.dot(np.sum(W, axis = 1)), (C, 1)).ravel()

    def hessian(P):
        W, b = decode(P)
        o = lambda x: np.outer(x, x)
        hess = sum([np.kron(np.diag(mu) - o(mu), o(X[i])) for i, mu in mus(W, b)])
        return hess + np.kron(np.eye(C), V0_inv)
    
    return objective, gradient, hessian

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

        # W = minimize(f2, [0] * ((D + 1) * C), method = 'Newton-CG', jac = g2, hess = H2).x
        loss, gradient, hessian = categorical_cross_entropy_loss(X, Y, penalty)
        W = minimize(loss, [0] * ((D + 1) * C)).x
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