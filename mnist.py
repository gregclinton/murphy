# https://www.youtube.com/watch?v=S75EdAcXHKk
# https://github.com/Newmu/Theano-Tutorials

import numpy as np
import gzip

def _read_bytes(name):
    with gzip.open('/home/greg/Downloads/' + name + '-ubyte.gz', 'rb') as f:
        return np.frombuffer(f.read(), dtype = np.uint8)
    
def _read_images(name, n):
    return _read_bytes(name)[16 :].reshape((n, 28 * 28)).astype(float) / 255.0

def _read_labels(name):
    bytes = _read_bytes(name)[8 :].flatten()
    o_h = np.zeros((len(bytes), 10))
    o_h[np.arange(len(bytes)), bytes] = 1
    return o_h

def load():
    trX = _read_images('train-images-idx3', 60000)
    teX = _read_images('t10k-images-idx3', 10000)
    trY = _read_labels('train-labels-idx1')
    teY = _read_labels('t10k-labels-idx1')
    return trX, teX, trY, teY