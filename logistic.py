import numpy as np
from scipy.optimize import minimize
from softmax import softmax, log_softmax
from scipy.special import expit
import tensorflow as tf
from sklearn import preprocessing 

class Classifier:
    '''
    logistic regression classifier
    murphy p. 255
    '''
    def __init__(self, C = 1):
        self.penalty = 0.0 / C
        
    def fit(self, X, y):
        N, D = X.shape
        C = len(np.unique(y))

        self.scaler = preprocessing.StandardScaler().fit(X)
        X = self.scaler.transform(X)
        
        if C == 2:
            ybar = np.mean(y)
            w0 = np.log(ybar / (1 - ybar))

            if False:
                y = y.reshape(N, 1)
                w = tf.Variable(tf.zeros([D, 1], dtype = tf.float64))
                eta = tf.matmul(X, w) + w0
                nll = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(eta, y))
                optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(nll)

                with tf.Session() as sess:
                    sess.run(tf.initialize_all_variables())
                    for i in range(1000):
                        sess.run(optimizer)
                    self.theta = w0, sess.run(w).ravel()            
            else:
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
            g0 = lambda W: sum([np.kron(mu - Y[i], X[i]) for i, mu in mus(W)])
            H0 = lambda W: sum([np.kron(np.diag(mu) - o(mu), o(X[i])) for i, mu in mus(W)])

            V0_inv = self.penalty * np.eye(D)
            
            f1 = lambda W: f0(W) + 0.5 * sum([w.dot(V0_inv).dot(w) for w in W.T])
            g1 = lambda W: g0(W) + np.tile(V0_inv.dot(np.sum(W, axis = 1)), (C, 1)).ravel()
            H1 = lambda W: H0(W) + np.kron(np.eye(C), V0_inv)

            fixup = lambda W: W.reshape(D, C)
            
            f2 = lambda W: f1(fixup(W))
            g2 = lambda W: g1(fixup(W))
            H2 = lambda W: H1(fixup(W))
                        
            W = minimize(f2, [0] * (D * C), method = 'Newton-CG', jac = g2, hess = H2).x
            self.theta = w0, fixup(W)
            
    def predict_log_proba(self, X):
        return np.log(self.predict_proba(X))

    def predict_proba(self, X):
        X = self.scaler.transform(X)
        w0, w = self.theta
        if w.ndim == 2:
            return softmax(X.dot(w))
        else:
            return expit(w0 + X.dot(w))

    def predict(self, X):
        p = self.predict_proba(X)
        return (p > 0.5) * 1 if len(p.shape) == 1 else np.argmax(p, axis = 1)