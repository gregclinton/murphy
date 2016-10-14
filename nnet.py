# https://keras.io/
# https://www.youtube.com/watch?v=S75EdAcXHKk
# https://github.com/Newmu/Theano-Tutorials

import numpy as np
import theano
import theano.tensor as T
from softmax import softmax

class Classifier:
    def __init__(self, H = 625, epochs = 1):
        self.H = H
        self.epochs = epochs

    def fit(self, X, Y):
        N, D = X.shape
        N, C = Y.shape
        H = self.H

        XX = T.fmatrix()
        YY = T.fmatrix()

        floatX = lambda X: np.asarray(X, dtype = theano.config.floatX)
        weights = lambda n, d: theano.shared(floatX(np.random.randn(n, d) * 0.01))
        
        w_h = weights(D, H)
        w_o = weights(H, C)
        h = T.nnet.sigmoid(T.dot(XX, w_h))

        py_x = T.nnet.softmax(T.dot(h, w_o))

        cost = T.mean(T.nnet.categorical_crossentropy(py_x, YY))
        params = [w_h, w_o]
        grads = T.grad(cost, params)
        learning_rate = 0.05
        updates = [[p, p - g * learning_rate] for p, g in zip(params, grads)]
        train = theano.function([XX, YY], updates = updates, allow_input_downcast = True)
        
        for _ in range(self.epochs):
            for start, end in zip(range(0, N, 128), range(128, N, 128)):
                train(X[start : end], Y[start : end])        
        self.theta = w_h.get_value(), w_o.get_value()

    def predict(self, X):
        w_h, w_o = self.theta
        return np.argmax(X.dot(w_h).dot(w_o), axis = 1)