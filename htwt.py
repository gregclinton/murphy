import pandas as pd

def get():
    url = 'https://raw.githubusercontent.com/probml/pmtk1/master/pmtk/data/heightWeightData.txt'
    data = pd.read_csv(url, header = None)

    X = data.ix[:, 1 : 2].values
    y = data.ix[:, 0].values
    y = (y == 1) * y
    return X, y    