import numpy as np
import sympy as sm
import tensorflow as tf
from tensorflow.python.framework.ops import Tensor
from tensorflow.python.ops import array_ops
import theano
import theano.tensor as T

def grad(fun, wrt = None):
    if isinstance(fun, Tensor):
        return tf.gradients(fun, wrt)[0]
    elif 'theano' in str(type(fun)):
        return T.grad(fun, [3.0, 1.0])
    elif 'sympy' in str(type(fun)):
        vars = list(fun.free_symbols)
        fns = [sm.lambdify(vars, sm.diff(fun, wrt)) for wrt in vars]

        def eval(x):
            x = np.array(x).astype(float)
            return [fn(*x) for fn in fns]
    return eval

def hess(fun, wrt = None):
    if isinstance(fun, Tensor):
        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/gradients.py
        gs = array_ops.unpack(tf.gradients(fun, wrt)[0])
        hess = [tf.gradients(g, wrt)[0] for g in gs]
        return array_ops.pack(hess)
    elif 'sympy' in str(type(fun)):
        vars = list(fun.free_symbols)
        n = len(vars)
        gs = [sm.diff(fun, wrt) for wrt in vars]
        fns = [sm.lambdify(vars, sm.diff(gs[i], wrt)) for i, wrt in enumerate(vars)]

        def eval(x):
            x = np.array(x).astype(float)
            return np.array([fns[i](*x) for i in xrange(n) for j in xrange(n)]).reshape(n, n)
    return eval