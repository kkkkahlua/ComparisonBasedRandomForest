# -*- coding:utf-8 -*-
import numpy as np
import time
from keras.datasets import mnist
from sklearn.model_selection import StratifiedKFold
from comp_rf import CompRF
from utils import calc_accuracy

if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train, y_train = X_train[:200], y_train[:200]
    X_test, y_test = X_test[:100], y_test[:100]

    n_train = X_train.shape[0]

    n0_list = [1, 4, 16, 64]
    M_list = [1, 4, 16, 64, 256]
    r = 1

    accuracy = -1
    n0_best = -1
    M_best = -1
    n_folds = 3     # requirement: 10
    classes = 10

    # train: 
    # 10-fold CV
    # n0 \in {1,4,16,64}
    # M \in {1,4,16,64,256}
    # get the best pair of (n, M)

    print('------------------now begin--------------------------')

    for n0 in n0_list:
        for M in M_list:
            cur_accuracy_list = []

            skf = StratifiedKFold(n_splits=n_folds)
            cv = [(t, v) for (t, v) in skf.split(range(n_train), y_train)]

            for k in range(n_folds):
                train_idx, val_idx = cv[k]

                comp_rf = CompRF(n0, M, r, "Classification", classes)

                time_start = time.time()
                y_predict = comp_rf.train_then_predict(X_train[train_idx], y_train[train_idx], X_train[val_idx])
                time_end = time.time()

                # comp_rf.fit_transform(X_train[train_idx], y_train[train_idx])
                # y_predict = comp_rf.predict(X_train[val_idx])
                cur_accuracy = calc_accuracy(y_train[val_idx], y_predict)

                cur_accuracy_list.append(cur_accuracy)

                print("(n0={0}, M={1}, fold={3}): {2:.2f}% [time={4:.2f}]".format(n0, M, cur_accuracy*100, k, time_end-time_start))

            cur_accuracy = sum(cur_accuracy_list) / len(cur_accuracy_list)
            print("(n0={0}, M={1}, average): {2:.2f}%".format(n0, M, cur_accuracy*100))
            if cur_accuracy > accuracy:
                accuracy = cur_accuracy
                n0_best = n0
                M_best = M
    
    print("best_accuracy={0}%, best_n_0={1}, best_M={2}".format(accuracy*100, n0_best, M_best))

    # test:
    comp_rf = CompRF(n0_best, M_best, r)
    y_predict = comp_rf.train_then_predict(X_train, y_train, X_test)
    # comp_rf.fit_transform(X_train, y_train)
    # y_predict = comp_rf.predict(X_test)
    accuracy = calc_accuracy(y_test, y_predict)

    print("accuracy: ", accuracy)