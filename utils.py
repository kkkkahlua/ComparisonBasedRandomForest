import numpy as np

def calc_accuracy(y_true, y_predict):
    return np.sum(y_true == y_predict) / len(y_true)

def vote_for_one(result):
    frequency = sorted([(np.sum(result == i), i) for i in set(result)])
    return frequency[-1][1]

def majority_vote(y_all):
    # y_all: n_test x num_of_predictor
    # y_predict: n_test
    n_test = y_all.shape[0]
    y_predict = np.zeros(n_test)
    for i in range(n_test):
        y_predict[i] = vote_for_one(y_all[i,:])
    return y_predict

def ratio_of_zero(v):
    return (len(v)-sum(v)) / len(v)

def confidence_level(y_proba):
    n_test = y_proba.shape[0]
    y_predict = np.zeros(n_test)
    for i in range(n_test):
        y_predict[i] = ratio_of_zero(y_proba[i,:])
    return y_predict

def euc_dist(x0, x1):
    x0 = x0.flatten(order='C')
    x1 = x1.flatten(order='C')
    return sum((x0-x1) ** 2)

def closer(x0, x1, x2):
    return euc_dist(x0, x1) <= euc_dist(x0, x2)

def highest_proba(proba):
    n_test = proba.shape[0]
    y_predict = np.zeros(n_test)
    for i in range(n_test):
        y_predict[i] = np.argmax(proba[i, :])
    return y_predict

def average(y):
    return sum(y) / len(y)