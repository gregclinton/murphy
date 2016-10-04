import pandas as pd
import numpy as np
import statsmodels.api as sm
import gzip

def iris():
    url = 'https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv'
    data = pd.read_csv(url)
    species = list(np.unique(data.species))
    y = np.array([species.index(s) for s in data.species])
    X = data.values[:, 0:-1]
    return X * 1.0, y * 1.0

def htwt():
    url = 'https://raw.githubusercontent.com/probml/pmtk1/master/pmtk/data/heightWeightData.txt'
    data = pd.read_csv(url, header = None)

    X = data.ix[:, 1 : 2].values
    y = data.ix[:, 0].values
    y = (y == 1) * y
    return X * 1.0, y * 1.0

def multiclass():
    mvn = stats.multivariate_normal.rvs
    y = 1.0 * np.random.choice(2, size = 200)
    X = np.array([mvn([1, 1] if heads else [4, 4]) for heads in y])
    return X, y

def murphy281():
    X = np.array([[2.2, 0.8], [2.6, 4.4], [1.4, 5.8], [2.4, 5.8], [3.4, 5.4], [3.4, 6.4],
        [5.5, 1.1], [5.6, 2.4], [5.5, 4.4], [6.4, 1.2], [6.4, 2.0], [7.8, 1.2], [7.6, 2.3]])
    y = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype = float)    
    return X, y

def passengers():
    url = 'https://vincentarelbundock.github.io/Rdatasets/csv/datasets/AirPassengers.csv'
    data = pd.read_csv(url).AirPassengers.values
    return pd.Series(data, pd.date_range('1949', '1960-12', freq = 'MS'))

def sunspots():
    data = sm.datasets.sunspots.load_pandas().data.SUNACTIVITY.values
    return pd.Series(data, pd.Index(pd.date_range('1700', '2008', freq = 'AS')))

def mnist():
    # https://www.youtube.com/watch?v=S75EdAcXHKk
    # https://github.com/Newmu/Theano-Tutorials
    
    def read_bytes(name):
        with gzip.open('/home/greg/Downloads/' + name + '-ubyte.gz', 'rb') as f:
            return np.frombuffer(f.read(), dtype = np.uint8)

    def read_images(name, n):
        return read_bytes(name)[16 :].reshape((n, 28 * 28)).astype(float) / 255.0

    def read_labels(name):
        bytes = read_bytes(name)[8 :].flatten()
        o_h = np.zeros((len(bytes), 10))
        o_h[np.arange(len(bytes)), bytes] = 1
        return o_h

    trX = read_images('train-images-idx3', 60000)
    teX = read_images('t10k-images-idx3', 10000)
    trY = read_labels('train-labels-idx1')
    teY = read_labels('t10k-labels-idx1')
    return trX, teX, trY, teY