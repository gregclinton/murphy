import numpy as np
from scipy.optimize import minimize
from sigmoid import sigmoid
from softmax import softmax, log_softmax

class Classifier:
    '''
    logistic regression classifier
    murphy p. 255
    '''
    def __init__(self, C = 1):
        self.penalty = 1.0 / C
        
    def preprocess(self, X):
        return X
        return X - self.mean
        return (X - self.mean) / np.sqrt(np.diag(self.cov))
        
    def fit(self, X, y):
        N, D = X.shape
        C = len(np.unique(y))

        X = np.array(X, dtype = float)
        self.mean = np.mean(X, axis = 0)
        self.cov = np.cov(X, ddof = 0, rowvar = False)
        
        X = self.preprocess(X)
        
        if C == 2:
            def nll(w):
                muw = mu(w)
                return -sum(y * np.log(muw) + (1 - y) * np.log(1 - muw))

            ybar = np.mean(y)
            w0 = np.log(ybar / (1 - ybar))
            mu = lambda w: sigmoid(w0 + X.dot(w))
            
            g = lambda w: X.T.dot(mu(w) - y)
            S = lambda w, mu: np.diag(mu * (1 - mu))
            H = lambda w: X.T.dot(S(w, mu(w))).dot(X)
            
            f_prime = lambda w: nll(w) + self.penalty * w.dot(w)
            g_prime = lambda w: g(w) + 2 * self.penalty * w
            H_prime = lambda w: H(w) + 2 * self.penalty * np.eye(D)
            
            w = minimize(f_prime, [0] * D, method = 'Newton-CG', jac = g_prime, hess = H_prime).x
            self.theta = w0, w
        else:
            Y = np.zeros((N, C))
            for i in range(N):
                Y[i, y[i]] = 1
            
            nll = lambda W: -sum([Y[i].dot(ll) for i, ll in enumerate(logmu(W))])
            mu = lambda W: softmax(X.dot(W))
            logmu = lambda W: log_softmax(X.dot(W))
            mus = lambda W: enumerate(mu(W))
            o = lambda x: np.outer(x, x)
            
            f0 = nll
            g0 = lambda W: np.sum([np.kron(mu - Y[i], X[i]) for i, mu in mus(W)])
            H0 = lambda W: np.sum([np.kron(np.diag(mu) - o(mu), o(X[i])) for i, mu in mus(W)])

            V0_inv = self.penalty * np.eye(D)
            
            f1 = lambda W: f0(W) + 0.5 * np.sum([w.dot(V0_inv).dot(w) for w in W.T])
            g1 = lambda W: g0(W) + V0_inv.dot(np.sum(W, axis = 0))
            H1 = lambda W: H0(W) + np.kron(np.eye(C), V0_inv)
            
            fixup = lambda W: W.reshape(D, C)
            
            f2 = lambda W: f1(fixup(W))
            g2 = lambda W: g0(fixup(W))
            H2 = lambda W: H1(fixup(W))
                        
            # W = minimize(f2, [0] * (D * C), method = 'Newton-CG', jac = g2, hess = H2).x
            W = minimize(f2, [0] * (D * C)).x
            w0 = 0
            self.theta = w0, fixup(W)
            
    def predict_log_proba(self, X):
        return np.log(self.predict_proba(X))

    def predict_proba(self, X):
        X = np.array(X, dtype = float)
        X = self.preprocess(X)
        w0, w = self.theta
        if w.ndim == 2:
            return softmax(X.dot(w))
        else:
            return sigmoid(w0 + X.dot(w))

    def predict(self, X):
        p = self.predict_proba(X)
        return (p > 0.5) * 1 if len(p.shape) == 1 else np.argmax(p, axis = 1)