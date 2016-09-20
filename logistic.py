import numpy as np
from scipy.optimize import minimize
from sigmoid import sigmoid
from softmax import softmax

class Classifier:
    '''
    logistic regression classifier
    murphy p. 255
    '''
    def __init__(self, C = 1):
        self.penalty = 1.0 / C
        
    def fit(self, X, y):
        N, D = X.shape
        C = len(np.unique(y))

        if C == 2:
            def NLL(w):
                muw = mu(w)
                return -sum(y * np.log(muw) + (1 - y) * np.log(1 - muw))

            ybar = np.mean(y)
            w0 = np.log(ybar / (1 - ybar))
            mu = lambda w: sigmoid(w0 + X.dot(w))
            
            g = lambda w: X.T.dot(mu(w) - y)
            S = lambda w, mu: np.diag(mu * (1 - mu))
            H = lambda w: X.T.dot(S(w, mu(w))).dot(X)
            
            f_prime = lambda w: NLL(w) + self.penalty * w.dot(w)
            g_prime = lambda w: g(w) + 2 * self.penalty * w
            H_prime = lambda w: H(w) + 2 * self.penalty * np.eye(D)
            
            w = np.zeros(D)
            w = minimize(f_prime, w, method = 'Newton-CG', jac = g_prime, hess = H_prime).x
            self.theta = w0, w
        else:
            def NLL(W):
                logs = np.log(mu(W))
                csum = lambda i: np.sum([y[i, c] * logs[i, c] for c in range(C)])
                csum = lambda i: y[i].dot(logs[i])
                return -sum([csum(i) for i in range(N)])
                return -sum(y.dot(np.log(mu(W))))

            mu = lambda W: softmax(X.dot(W))
            o = lambda x: np.outer(x, x)
            
            g = lambda W: np.sum([np.kron(mui - y[i], x[i]) for i, mui in enumerate(mu(W))])
            H = lambda W: np.sum([np.kron(diag(mui) - o(mui), o(x[i])) for i, mui in enumerate(mu(W))])

            V0_inv = penalty * np.eye(C)
            
            f_prime = lambda w: NLL(w) + 0.5 * np.sum([w.dot(V0_inv).dot(w) for w in W.T])
            g_prime = lambda w: g(w) + V0_inv.dot(np.sum(W, axis = 1))
            H_prime = lambda w: H(w) + np.kron(np.eye(C), V0_inv)
            
            W = np.zeros((D, C))
            W = minimize(f_prime, W, method = 'Newton-CG', jac = g_prime, hess = H_prime).x
            self.theta = W

    def predict_log_proba(self, X):
        return np.log(self.predict_proba(X))

    def predict_proba(self, X):
        if len(self.theta) == 1:
            W = self.theta
            return softmax(X.dot(W))
        else:
            w0, w = self.theta
            return sigmoid(w0 + X.dot(w))

    def predict(self, X):
        p = self.predict_proba(X)
        return (p > 0.5) * 1 if len(p.shape) == 1 else np.argmax(p, axis = 1)