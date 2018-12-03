# -*- coding:utf-8 -*-
from tree_node import TreeNode
import random
import numpy as np
from utils import closer
from utils import vote_for_one
from utils import average

class CompTree(object):
    def __init__(self, n0):
        self.n0 = n0
        self.root = None
        self.task = "classification"
        self.depth = -1
        self.X = None
        self.y = None



    # def build_tree(self, X, y, depth):
    #     if X.shape[0] <= self.n0:
    #         self.depth = max(depth, depth)
    #         return TreeNode(y, None, None, None, None, depth)

    #     x0_idx, x1_idx = self.pick_two_samples(y)
    #     x0 = X[x0_idx]
    #     x1 = X[x1_idx]

    #     left_idx, right_idx = self.separate_samples(X, x0, x1)

    #     left_child = self.build_tree(X[left_idx], y[left_idx], depth+1)
    #     right_child = self.build_tree(X[right_idx], y[right_idx], depth+1)

    #     return TreeNode(None, x0, x1, left_child, right_child, depth)

    # def pick_two_samples(self, y):
    #     n_data = y.shape[0]

    #     left_idx = [i for i in range(n_data) if y[i] == y[0]]
    #     right_idx = [i for i in range(n_data) if i not in left_idx]

    #     assert len(left_idx) + len(right_idx) == n_data

    #     if not left_idx:
    #         idx = random.sample(range(len(right_idx)), 2)
    #         x0_idx = right_idx[idx[0]]
    #         x1_idx = right_idx[idx[1]]
    #     elif not right_idx:
    #         idx = random.sample(range(len(left_idx)), 2)
    #         x0_idx = left_idx[idx[0]]
    #         x1_idx = left_idx[idx[1]]
    #     else:
    #         x0_idx = left_idx[random.randint(0, len(left_idx)-1)]
    #         x1_idx = right_idx[random.randint(0, len(right_idx)-1)]
    #     return x0_idx, x1_idx

    def pick_two_samples(self, indice):
        n_data = len(indice)
        indice_0 = [idx for i, idx in enumerate(indice) if self.y[idx] == self.y[0]]
        indice_1 = [idx for i, idx in enumerate(indice) if idx not in indice_0]
        assert len(indice_0) + len(indice_1) == n_data
        
        if not indice_0:
            idxes = random.sample(range(len(indice_1)), 2)
            idx_0, idx_1 = indice_1[idxes[0]], indice_1[idxes[1]]
        elif not indice_1:
            idxes = random.sample(range(len(indice_0)), 2)
            idx_0, idx_1 = indice_0[idxes[0]], indice_0[idxes[1]]
        else:
            idx_0 = indice_0[random.randint(0, len(indice_0)-1)]
            idx_1 = indice_1[random.randint(0, len(indice_1)-1)]
        return idx_0, idx_1

    # def separate_samples(self, X, x0, x1):
    #     left_idx = [idx for idx, x in enumerate(X) if euc_dist(x, x0) <= euc_dist(x, x1)]
    #     right_idx = [idx for idx in range(X.shape[0]) if idx not in left_idx]
    #     return left_idx, right_idx

    def separate_samples(self, indice, idx_0, idx_1):
        left_idx = [idx for i, idx in enumerate(indice) if closer(self.X[idx], self.X[idx_0], self.X[idx_1])]
        right_idx = [idx for i, idx in enumerate(indice) if idx not in left_idx]
        return left_idx, right_idx

    def build_tree(self, indice, depth):
        if len(indice) <= self.n0:
            self.depth = max(depth, self.depth)
            if self.task == "classfication":
                value = vote_for_one(self.y[indice])
            else:
                value = average(self.y[indice])
            return TreeNode(value, None, None, None, None, depth)
        
        idx_0, idx_1 = self.pick_two_samples(indice)
        left_indice, right_indice = self.separate_samples(indice, idx_0, idx_1)

        left_child = self.build_tree(left_indice, depth+1)
        right_child = self.build_tree(right_indice, depth+1)
        return TreeNode(None, idx_0, idx_1, left_child, right_child, depth)

    def fit_transform(self, X, y):
        self.X = X
        self.y = y
        self.root = self.build_tree(range(X.shape[0]), 0)
        # self.root = self.build_tree(X, y, 0)

    def trace_in_tree(self, x):
        node = self.root
        while node.left_child is not None:
            if closer(x, self.X[node.left_pivot], self.X[node.right_pivot]):
                node = node.left_child
            else:
                node = node.right_child
        return node.value
    
    def vote(self, leaf_node):
        return vote_for_one(self.y[leaf_node.indice])

    def average(self, leaf_node):
        y = self.y[leaf_node.indice]
        return sum(y) / len(y)

    def predict(self, X_test):
        y_predict = np.zeros(X_test.shape[0])
        for i, x in enumerate(X_test):
            y_predict[i] = self.trace_in_tree(x)
        return y_predict