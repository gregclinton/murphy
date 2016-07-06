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

def train(X, Y, cost, updates, trX, teX, trY, teY):
    train = theano.function(inputs = [X, Y], outputs = outputs, updates = updates, allow_input_downcast = True)
    predict = theano.function(inputs = [X], outputs = outputs, allow_input_downcast = True)

    for i in range(100):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
            train(trX[start : end], trY[start : end])
            
        sys.stdout.write(np.mean(np.argmax(teY, axis = 1) == predict(teX)))
        sys.stdout.write(' ')
        sys.stdout.flush()