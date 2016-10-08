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