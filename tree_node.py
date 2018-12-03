# -*- coding:utf-8 -*-

class TreeNode(object):
    def __init__(self, value, left_pivot=None, right_pivot=None, left_child=None, right_child=None, depth=-1):
        self.value = value
        self.left_pivot = left_pivot
        self.right_pivot = right_pivot
        self.left_child = left_child
        self.right_child = right_child
        self.depth = depth
