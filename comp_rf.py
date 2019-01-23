# -*- coding:utf-8 -*-
from comp_tree import CompTree
from utils import majority_vote
from utils import confidence_level
from utils import highest_proba
import collections
import numpy as np
import time

class CompRF(object):
    def __init__(self, n0, M, r, task=None, classes=None):
        self.n0 = n0
        self.M = M
        self.r = r
        self.class_no = 2   # default: 2 classes
        self.task = task
        self.classes = classes

    def build_forest(self, X_train, y_train, X_test, kind):
        n_test = X_test.shape[0]
        result = np.zeros((n_test, self.M))        # the prediction of M trees, n_data x M
        if (kind == "binaryclass"):    
            for i in range(self.M):
                comp_tree = CompTree(self.n0, kind)
                time_start = time.clock()
                comp_tree.fit_transform(X_train, y_train)
                time_end = time.clock()
                print(time_end - time_start)
                result[:, i] = comp_tree.predict(X_test)
        return result

    def calc_class_no(self, y):
        y_sorted = y
        y_sorted.sort()
        # TODO:
        counter = collections.Counter()
        # print(counter)
        return len(counter)

    def train_then_predict(self, X_train, y_train, X_test):
        if (self.task == "Classification"):
            self.class_no = self.calc_class_no(y_train)
            
            label_set = set(y_train)

            n_classes = len(label_set)
            n_test = X_test.shape[0]
            n_ovo = int(n_classes * (n_classes - 1) / 2)
            n_ovr = n_classes

            class_data = [[] for i in range(n_classes)]
            for idx, y in enumerate(y_train):
                class_data[y].append(idx)

            tickets = np.zeros((n_test, n_ovo))     # n_test x n_ovo
            proba = np.zeros((n_test, n_ovr))       # n_test x n_ovr
            # ovo
            # cnt = 0
            # for class_i in range(n_classes):
            #     for class_j in range(class_i+1, n_classes):
            #         X_train_ij = np.vstack((X_train[class_data[class_i]], X_train[class_data[class_j]]))
            #         y_train_ij = np.concatenate((y_train[class_data[class_i]], y_train[class_data[class_j]]))

            #         result = self.build_forest(X_train_ij, y_train_ij, X_test, kin="binaryclass")  # prediction result of one forest

            #         tickets[:, cnt] = majority_vote(result)       # n_test x 1
            #         cnt = cnt + 1
            
            # return majority_vote(tickets)

            # ovr
            cnt = 0
            for class_0_idx in range(n_classes):
                X_train_0 = X_train[class_data[class_0_idx]]
                X_train_1_indice = []
                
                time_start = time.clock()
                for i in range(n_classes):
                    if i != class_0_idx:
                        X_train_1_indice = X_train_1_indice + class_data[i]
                time_end = time.clock()
                time_1 = time_end - time_start

                X_train_1 = X_train[X_train_1_indice]
                y_train_0 = np.zeros(X_train_0.shape[0])
                y_train_1 = np.ones(X_train_1.shape[0])
                X_train_01 = np.vstack((X_train_0, X_train_1))
                y_train_01 = np.concatenate((y_train_0, y_train_1))
                result = self.build_forest(X_train_01, y_train_01, X_test, kind="binaryclass")  # prediction result of one forest, in the form of label, n_data x M

                proba[:, cnt] = confidence_level(result)
                cnt = cnt + 1

            return highest_proba(proba)
