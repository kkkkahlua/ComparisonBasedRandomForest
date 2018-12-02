# -*- coding:utf-8 -*-
from comp_tree import CompTree
from utils import majority_vote
import numpy as np

class CompRF(object):
    def __init__(self, n0, M, r, task=None, classes=None):
        self.n0 = n0
        self.M = M
        self.r = r
        self.task = task
        self.classes = classes

    def build_forest(self, X_train, y_train, X_test):
        n_test = X_test.shape[0]
        result = np.zeros((n_test, self.M))        # the prediction of M trees, n_data x M
        for i in range(self.M):
            comp_tree = CompTree(self.n0)
            comp_tree.fit_transform(X_train, y_train)
            result[:, i] = comp_tree.predict(X_test)
        return result

    def train_then_predict(self, X_train, y_train, X_test):
        if (self.task == "Classification"):
            label_set = set(y_train)

            n_classes = len(label_set)
            n_test = X_test.shape[0]
            n_ovo = int(n_classes * (n_classes - 1) / 2)

            class_data = [[] for i in range(n_classes)]
            for idx, y in enumerate(y_train):
                class_data[y].append(idx)

            tickets = np.zeros((n_test, n_ovo))     # n_test x n_ovo
            cnt = 0
            for class_i in range(n_classes):
                for class_j in range(class_i+1, n_classes):
                    X_train_ij = np.vstack((X_train[class_data[class_i]], X_train[class_data[class_j]]))
                    y_train_ij = np.concatenate((y_train[class_data[class_i]], y_train[class_data[class_j]]))

                    result = self.build_forest(X_train_ij, y_train_ij, X_test)
                    tickets[:, cnt] = majority_vote(result)       # n_test x 1
                    cnt = cnt + 1
            
            return majority_vote(tickets)