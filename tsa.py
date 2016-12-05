import numpy as np

# see https://github.com/statsmodels/statsmodels/blob/master/statsmodels/tsa/seasonal.py

def decompose(x, freq):
    n = len(x)
    filt = np.array([.5] + [1] * (freq - 1) + [.5]) / freq
    a = [np.sum(x[i : i + len(filt)] * filt) for i in range(len(x) - freq)]
    pad = [np.nan] * (freq // 2)
    trend = np.array(pad + a + pad)
    detrended = x - trend
    avgs = np.array([np.nanmean(detrended[i::freq]) for i in range(freq)])
    avgs -= np.mean(avgs)
    seasonal = np.tile(avgs, n // freq + 1)[:n]
    resid = detrended - seasonal
    return trend, seasonal, resid

def acf(x, nlags):
    xbar = np.mean(x)
    n = len(x)
    boo = lambda t, k: (x[t] - xbar) * (x[t + k] - xbar)
    c = lambda k: np.sum([boo(t, k) for t in xrange(n - k)]) / n
    return np.array([c(k) for k in xrange(nlags + 1)]) / c(0)

def ccf(x1, x2, nlags):
    x = [x1, x2]
    xbar = [np.mean(x1), np.mean(x2)]
    n = len(x1)
    boo = lambda t, k, i, j: (x[i][t + k] - xbar[i]) * (x[j][t] - xbar[j])
    c = lambda k, i, j: np.sum([boo(t, k, i, j) for t in xrange(n - k)]) / n
    ccvf = np.array([c(k, 0, 1) for k in xrange(nlags + 1)])
    return  ccvf / np.sqrt(c(0, 0, 0) * c(0, 1, 1))

def ewma(x, alpha = None, com = None):   
    alpha = alpha or 1.0 / (1 + com)
    a = []
    
    for i in range(len(x)):
        a.append(x[i] if i == 0 else alpha * x[i] + (1 - alpha) * a[i - 1])
            
    return np.array(a)