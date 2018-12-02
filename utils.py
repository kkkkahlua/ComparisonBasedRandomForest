import numpy as np

def CalcAccuracy(y_true, y_predict):
    return np.sum(y_true == y_predict) / len(y_true)

def MajorityVote(result):
    # result: M x n_data
    # ret: n_data
    # TODO: rewrite
    frequency = [np.sum(np.array(result) == i, i) for i in set(result)]
    return frequency[-1][1]

def euc_dist(x0, x1):
    x0 = x0.flatten(order='C')
    x1 = x1.flatten(order='C')
    return sum((x0-x1) ** 2)