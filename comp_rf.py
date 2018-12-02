# -*- coding:utf-8 -*-
from comp_tree import CompTree
from utils import MajorityVote
import numpy as np

class CompRF(object):
    def __init__(self, n0, M, r, task=None, classes=None):
        self.n0 = n0
        self.M = M
        self.r = r
        self.task = task

    def predict_after_train(self, X_train, y_train, X_test):
        if (self.task == "Classification"):
            label_set = set(y_train)

            n_classes = len(label_set)
            n_data = X_train.shape[0]
            n_ovo = int(n_classes * (n_classes - 1) / 2)

            class_data = [[] for i in range(n_classes)]
            for idx, y in enumerate(y_train):
                class_data[y].append(idx)

            tickets = np.zeros((n_data, n_ovo))     # n_data x n_ovo
            cnt = 0
            for class_i in range(n_classes):
                for class_j in range(n_classes):
                    X_train_ij = np.vstack((X_train[class_data[class_i]], X_train[class_data[class_j]]))
                    y_train_ij = np.vstack((y_train[class_data[class_i]], y_train[class_data[class_j]]))

                    result = np.zeros((n_data, self.M))        # the prediction of M trees, n_data x M
                    for i in range(self.M):
                        comp_tree = CompTree(self.n0)
                        comp_tree.fit_transform(X_train_ij, y_train_ij)
                        result[:, i] = comp_tree.predict(X_test) 

                    tickets[:, cnt] = MajorityVote(result)       # (n_data x 1)
                    cnt = cnt + 1
            
            return MajorityVote(tickets)


    # def fit_transform(self, X_train, y_train):
    #     for i in range(M):
    #         comp_tree = CompTree(n0)
    #         comp_tree.fit_transform(X_train)
    #         predictor.append(comp_tree)

    # def predict(self, X_test):
    #     pass