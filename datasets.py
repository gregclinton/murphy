import pandas as pd
import numpy as np
import statsmodels.api as sm
import gzip
import requests
import json

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
    return pd.Series(data, pd.period_range('1949', periods = len(data), freq = 'M'))

def sunspots():
    data = sm.datasets.sunspots.load_pandas().data.SUNACTIVITY.values
    return pd.Series(data, pd.Index(pd.date_range('1700', '2008', freq = 'AS')))

cowpertwait = 'https://raw.githubusercontent.com/burakbayramli/kod/master/books/Introductory_Time_Series_with_R_Metcalfe/'

def maine():
    data = pd.read_csv(cowpertwait + 'Maine.dat').unemploy.values
    return pd.Series(data, pd.Index(pd.period_range('1996', periods = len(data), freq = 'M')))

def exchange():
    data = pd.read_csv(cowpertwait + 'pounds_nz.dat').xrate.values
    return pd.Series(data, pd.Index(pd.period_range('1991', periods = len(data), freq = 'Q')))

def cbe(col):
    data = pd.read_table(cowpertwait + 'cbe.dat', sep = '\t').values[:, col]
    return pd.Series(data, pd.Index(pd.period_range('1958', periods = len(data), freq = 'M')))

def choc():
    return cbe(0)

def beer():
    return cbe(1)

def elec():
    return cbe(2)

def warming():
    url = 'https://crudata.uea.ac.uk/cru/data/temperature/HadCRUT4-gl.dat'
    df = pd.read_fwf(url, header = None)
    data = df.iloc[::2, 1 :-1].values.ravel()
    return pd.Series(data, pd.Index(pd.period_range('1850', periods = len(data), freq = 'M')))

def yql(q):
    yql = 'https://query.yahooapis.com/v1/public/yql'
    url = '%s?q=%s&format=json' % (yql, q)
    o = json.loads(requests.get(url).text)
    return o['query']['results']    

def stock(symbol):
    q = "env 'store://datatables.org/alltableswithkeys' ; "
    q += "select Ask from yahoo.finance.quotes where symbol = '%s' " % symbol
    return yql(q)

def stock(symbol, start, end):
    q = "env 'store://datatables.org/alltableswithkeys' ; "
    q += "select Date, Close from yahoo.finance.historicaldata "
    q += "where symbol = '%s' and startDate = '%s' and endDate = '%s'" % (symbol, start, end)
    df = pd.DataFrame(yql(q)['quote'])
    series = pd.Series(df.Close.values, pd.PeriodIndex(df.Date, freq = 'D'))
    return series.sort_index().astype(float)

def noaa(datasetid, zipcode, start, end):
    # http://www.ncdc.noaa.gov/cdo-web/webservices/v2
    # http://www1.ncdc.noaa.gov/pub/data/cdo/documentation/GHCND_documentation.pdf
    # see TMAX
    # http://www1.ncdc.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt
    token = 'nvPClxSghOlFavUKyLzkOmzUaIcqRrfN'
    headers = {'Content-type': 'application/json', 'token': token}
    noaa = 'http://www.ncdc.noaa.gov/cdo-web/api/v2/data'
    q = 'datasetid=%s&locationid=ZIP:%s&startdate=%s&enddate=%s' % (datasetid, zipcode, start, end)
    url = '%s?%s' % (noaa, q)
    o = requests.get(url, headers = headers)
    o = json.loads(o.text)
    return pd.DataFrame(o['results'])
    
def temperature(zipcode, start, end):
    return noaa('GHCND', zipcode, start, end)
    
def bls(seriesid, start, end):
    # bureau of labor statistics
    # http://www.bls.gov/developers/
    headers = {'Content-type': 'application/json'}
    url = 'http://api.bls.gov/publicAPI/v2/timeseries/data/'
    q = {}
    q['seriesid'] = [seriesid]
    q['startyear'] = start
    q['endyear'] = end
    q['registrationKey'] = '9de5607186434ebb84520bf1120fc943'
    o = requests.post(url, data = json.dumps(q), headers = headers)
    o = json.loads(o.text)
    return o['Results']['series'][0]['data']
    
def unemployment(start, end):
    o = bls('LNS14000000', start, end)
    o = [(row['year'] + '-' + row['period'][1 : 3], row['value']) for row in o]
    o = np.array(o)
    ts = pd.Series(o[:, 1], pd.PeriodIndex(o[:, 0], freq = 'M'))
    return ts.sort_index().astype(float)

def cpi(start, end):
    # consumer price index (used to calculate inflation)
    o = bls('CUUR0000SA0L1E', start, end)
    o = [(row['year'] + '-' + row['period'][1 : 3], row['value']) for row in o]
    o = np.array(o)
    data = o[:, 1]
    ts = pd.Series(o[:, 1], pd.PeriodIndex(o[:, 0], freq = 'M'))
    return ts.sort_index().astype(float)

def inflation(start, end):
    x = cpi(start - 1, end)
    before = x.iloc[:-12].values
    after = x.iloc[12:].values
    data = np.round(100. * (after - before) / before, 1)
    return pd.Series(data, pd.period_range(start, periods = len(data), freq = 'M'))

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