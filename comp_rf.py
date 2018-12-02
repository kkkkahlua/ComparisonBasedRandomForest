# -*- coding:utf-8 -*-
from comp_tree import CompTree
from utils import MajorityVote

class CompRF(object):
    def __init__(self, n0, M, r):
        self.n0 = n0
        self.M = M
        self.r = r
        self.result = []

    def predict_after_train(self, X_train, y_train, X_test):
        for i in range(self.M):
            comp_tree = CompTree(self.n0)
            comp_tree.fit_transform(X_train, y_train)
            self.result.append(comp_tree.predict(X_test))
        
        return MajorityVote(self.result)

    # def fit_transform(self, X_train, y_train):
    #     for i in range(M):
    #         comp_tree = CompTree(n0)
    #         comp_tree.fit_transform(X_train)
    #         predictor.append(comp_tree)

    # def predict(self, X_test):
    #     pass