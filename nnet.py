# https://keras.io/
# https://www.youtube.com/watch?v=S75EdAcXHKk
# https://github.com/Newmu/Theano-Tutorials

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
import sys

srng = RandomStreams()

def floatX(X):
    return np.asarray(X, dtype = theano.config.floatX)

def weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def rectify(X):
    return T.maximum(X, 0.0)

def softmax(X):
    e_x = T.exp(X - X.max(axis = 1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis = 1).dimshuffle(0, 'x')

def dropout(X, p):
    if (p > 0):
        X *= srng.binomial(X.shape, p = 1 - p, dtype = theano.config.floatX)
        return X / (1 - p)
    else:
        return X
    
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

        w_h = weights((D, H))
        w_o = weights((H, C))
        h = T.nnet.sigmoid(T.dot(XX, w_h))

        py_x = T.nnet.softmax(T.dot(h, w_o))
        y_x = T.argmax(py_x, axis = 1)

        cost = T.mean(T.nnet.categorical_crossentropy(py_x, YY))
        params = [w_h, w_o]
        updates = sgd(cost, params)        
        theano.function([XX, YY], [], updates = updates, allow_input_downcast = True)(X, Y)
        self.theta = w_h, w_o

    def predict(self, X):
        w_h, w_o = self.theta