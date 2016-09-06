import numpy as np

def mcc(y, y_hat):
    y = np.array(y)
    y_hat = np.array(y_hat)
    TP = sum((y == 1) & (y_hat == 1))
    FP = sum((y == 0) & (y_hat == 1))
    TN = sum((y == 0) & (y_hat == 0))
    FN = sum((y == 1) & (y_hat == 0))
    d = np.sqrt(np.exp(sum(np.log([(TP + FP), (TP + FN), (TN + FP), (TN + FN)]))))
    return ((TP * TN) - (FP * FN)) / d