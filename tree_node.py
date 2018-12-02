# -*- coding:utf-8 -*-

class TreeNode(object):
    def __init__(self, X, y, left_pivot=None, right_pivot=None, left_child=None, right_child=None):
        self.X = X
        self.y = y
        self.left_pivot = left_pivot
        self.right_pivot = right_pivot
        self.left_child = None
        self.right_child = None
        