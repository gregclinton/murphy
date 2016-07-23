import pandas as pd

url = 'https://raw.githubusercontent.com/probml/pmtk3/master/bigData/14cancer/14cancer.'

def get_train():
    X = pd.read_fwf(url + 'xtrain', header = None, skiprows = 1).values.T
    y = pd.read_table(url + 'ytrain', sep = ' ', header = None).ix[0, 1:].values
    return X, y.astype(int) - 1

def get_test():
    X = pd.read_fwf(url + 'xtest', header = None, skiprows = 1).values.T
    y = pd.read_table(url + 'ytest', sep = ' ', header = None, skiprows = 1).ix[0, 1:].values
    return X, y.astype(int) - 1