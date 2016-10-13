# https://keras.io/
# https://www.youtube.com/watch?v=S75EdAcXHKk
# https://github.com/Newmu/Theano-Tutorials

import numpy as np
import theano.tensor as T
from softmax import softmax

sgd_update = lambda p, g, lr: p - lr * g

def sgd(cost, params, lr = 0.05):
    grads = T.grad(cost, params)
    updates = []
    for p, g in zip(params, grads):
        updates.append([p, p - g * lr])
    return updates    

def rms(cost, params, lr = 0.001, rho = 0.9, epsilon = 1e-6):
    grads = T.grad(cost, params)
    updates = []

    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g /= gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))

    return updates

class Classifier:
    def fit(self, X, Y):
        N, D = X.shape
        N, C = Y.shape
        H = 250

        XX = T.fmatrix()
        YY = T.fmatrix()

        floatX = lambda X: np.asarray(X, dtype = theano.config.floatX)
        weights = lambda n, d: theano.shared(floatX(np.random.randn(n, d) * 0.01))
        
        w_h = weights(D, H)
        w_o = weights(H, C)
        h = T.nnet.sigmoid(T.dot(XX, w_h))

        py_x = T.nnet.softmax(T.dot(h, w_o))
        y_x = T.argmax(py_x, axis = 1)

        cost = T.mean(T.nnet.categorical_crossentropy(py_x, YY))
        params = [w_h, w_o]
        updates = sgd(cost, params)
        train = theano.function([XX, YY], [], updates = updates, allow_input_downcast = True)
        
        for i in range(1):
            for start, end in zip(range(0, N, 128), range(128, N, 128)):
                train(X[start : end], Y[start : end])        
        self.theta = w_h, w_o

    def predict(self, X):
        w_h, w_o = self.theta
        w_h = w_h.get_value()
        w_o = w_o.get_value()
        return np.argmax(X.dot(w_h).dot(w_o), axis = 1)