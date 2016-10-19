import numpy as np
from scipy.optimize import minimize
from softmax import softmax, log_softmax
from scipy.special import expit
from sklearn import preprocessing 

class CategoricalCrossEntropyLoss:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.penalty = 0.0

    def mus(W, b):
        return enumerate(softmax(self.X.dot(W, b)))

    def parts(self, P):
        N, D = self.X.shape
        N, C = self.Y.shape
        W, b = P[:-D].reshape(D, C), P[-D:]
        V0_inv = self.penalty * np.eye(D)
        return W, b, C, V0_inv

    def objective(self, P):
        W, b, C, V0_inv = self.parts(P)
        logmu = lambda W, b: log_softmax(X.dot(W, b))
        nll = lambda W, b: -sum([Y[i].dot(ll) for i, ll in enumerate(logmu(W, b))])
        f0 = nll
        penalty =  0.5 * sum([w.dot(V0_inv).dot(w) for w in W.T])
        return f0(W, b) + penalty

    def gradient(self, P):
        W, b, C, V0_inv = self.parts(P)
        g0 = lambda W, b: sum([np.kron(mu - Y[i], X[i]) for i, mu in mus(W, b)])
        penalty = np.tile(V0_inv.dot(np.sum(W, axis = 1)), (C, 1))
        return (g0(W, b) + penalty).ravel()

    def hessian(self, P):
        W, b, C, V0_inv = self.parts(P)
        o = lambda x: np.outer(x, x)
        H0 = lambda W, b: sum([np.kron(np.diag(mu) - o(mu), o(X[i])) for i, mu in mus(W, b)])
        penalty = np.kron(np.eye(C), V0_inv)
        return H0(W, b) + penalty

def one_hot(y):
    N, C = len(y), len(np.unique(y))
    Y = np.zeros((N, C))
    for i in range(N):
        Y[i, y[i]] = 1
    return Y
    
class Classifier:
    '''
    logistic regression classifier
    murphy p. 255
    '''
    def __init__(self):
        self.penalty = 0.0
        
    def fit(self, X, y):
        N, D = X.shape
        C = len(np.unique(y))

        # self.scaler = preprocessing.StandardScaler().fit(X)
        # X = self.scaler.transform(X)
        ybar = np.mean(y)
        w0 = np.log(ybar / (1 - ybar))
        
        if C == 2:
            def nll(w):
                muw = mu(w)
                return -sum(y * np.log(muw) + (1 - y) * np.log(1 - muw))

            mu = lambda w: expit(w0 + X.dot(w))
            S = lambda w, mu: np.diag(mu * (1 - mu))

            f0 = nll
            g0 = lambda w: X.T.dot(mu(w) - y)
            H0 = lambda w: X.T.dot(S(w, mu(w))).dot(X)

            f1 = lambda w: f0(w) + self.penalty * w.dot(w)
            g1 = lambda w: g0(w) + 2 * self.penalty * w
            H1 = lambda w: H0(w) + 2 * self.penalty * np.eye(D)

            w = minimize(f1, [0] * D, method = 'Newton-CG', jac = g1, hess = H1).x
            self.theta = w, w0
        else:
            Y = one_hot(y)
            w0 = np.zeros(C)

            eta = lambda W, b: X.dot(W) + np.tile(b, (N, 1))
            
            mu = lambda W, b: softmax(eta(W, b))
            logmu = lambda W, b: log_softmax(eta(W, b))
            mus = lambda W, b: enumerate(mu(W, b))

            f0 = nll = lambda W, b: -sum([Y[i].dot(ll) for i, ll in enumerate(logmu(W, b))])
            V0_inv = self.penalty * np.eye(D)
            f1 = lambda W, b: f0(W, b) + 0.5 * sum([w.dot(V0_inv).dot(w) for w in W.T])
            
            def f2(P):
                W, b = P[:-C].reshape(D, C), P[-C:]
                return f1(W, b)

            # W = minimize(f2, [0] * ((D + 1) * C), method = 'Newton-CG', jac = g2, hess = H2).x
            W = minimize(f2, [0] * ((D + 1) * C)).x
            self.theta = W[:-C].reshape(D, C), W[-C:]

    def predict_log_proba(self, X):
        return np.log(self.predict_proba(X))

    def predict_proba(self, X):
        # X = self.scaler.transform(X)
        N, D = X.shape
        w, w0 = self.theta
        if w.ndim == 2:
            return softmax(X.dot(w) + np.tile(w0, (N, 1)))
        else:
            return expit(X.dot(w) + w0)

    def predict(self, X):
        p = self.predict_proba(X)
        return (p > 0.5) * 1 if len(p.shape) == 1 else np.argmax(p, axis = 1)