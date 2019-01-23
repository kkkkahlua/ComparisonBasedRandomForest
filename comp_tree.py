# -*- coding:utf-8 -*-
from tree_node import TreeNode
import random
import numpy as np
from utils import closer
from utils import vote_for_one
from utils import average

class CompTree(object):
    def __init__(self, n0, kind):
        self.n0 = n0
        self.root = None
        self.task = "classification"
        self.depth = -1
        self.X = None
        self.y = None
        self.kind = kind
        self.leaf = 0

    def pick_two_samples(self, indice_0, indice_1):
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

    def separate_samples(self, indice_0, indice_1, idx_0, idx_1):
        left_idx_0, right_idx_0, left_idx_1, right_idx_1 = [], [], [], []
        for idx in indice_0:
            if closer(self.X[idx], self.X[idx_0], self.X[idx_1]):
                left_idx_0.append(idx)
            else:
                right_idx_0.append(idx)
        for idx in indice_1:
            if closer(self.X[idx], self.X[idx_0], self.X[idx_1]):
                left_idx_1.append(idx)
            else:
                right_idx_1.append(idx)
        # left_idx_0 = [idx for i, idx in enumerate(indice_0) if closer(self.X[idx], self.X[idx_0], self.X[idx_1])]
        # right_idx_0 = [idx for i, idx in enumerate(indice_0) if idx not in left_idx_0]
        # left_idx_1 = [idx for i, idx in enumerate(indice_1) if closer(self.X[idx], self.X[idx_0], self.X[idx_1])]
        # right_idx_1 = [idx for i, idx in enumerate(indice_1) if idx not in left_idx_1]
        # print(left_idx_0, len(left_idx_0))
        # print(left_idx_1, len(left_idx_1))
        # print(right_idx_0, len(right_idx_0))
        # print(right_idx_1, len(right_idx_1))
        return left_idx_0, left_idx_1, right_idx_0, right_idx_1

    def build_binary_tree(self, indice_0, indice_1, depth):
        self.depth = depth

        if len(indice_0) + len(indice_1) <= self.n0:
            # print(self.depth)
            if self.task == "classfication":
                value = vote_for_one(self.y[indice_0 + indice_1])
            else:
                value = average(self.y[indice_0 + indice_1])
            return TreeNode(value, None, None, None, None, depth)
        
        idx_0, idx_1 = self.pick_two_samples(indice_0, indice_1)
        left_indice_0, left_indice_1, right_indice_0, right_indice_1 = self.separate_samples(indice_0, indice_1, idx_0, idx_1)
        left_child = self.build_binary_tree(left_indice_0, left_indice_1, depth+1)
        right_child = self.build_binary_tree(right_indice_0, right_indice_1, depth+1)
        return TreeNode(None, idx_0, idx_1, left_child, right_child, depth)

    def fit_transform(self, X, y):
        self.X = X
        self.y = y

        # print(y, len(y))

        indice = range(X.shape[0])
        n_data = len(indice)
        # print('n_data = {0}'.format(n_data))
        indice_0 = [idx for i, idx in enumerate(indice) if self.y[idx] == self.y[0]]
        indice_1 = [idx for i, idx in enumerate(indice) if idx not in indice_0]
        assert len(indice_0) + len(indice_1) == n_data

        # print(indice_0, indice_1, len(indice_0), len(indice_1))

        self.root = self.build_binary_tree(indice_0, indice_1, 0)
        self.output_tree(self.root)

    def output_tree(self, root):
        if root is None:
            return
        if root.left_child is None and root.right_child is None:
            self.leaf = self.leaf + 1
            # print(root.depth, self.leaf)
        self.output_tree(root.left_child)
        self.output_tree(root.right_child)

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