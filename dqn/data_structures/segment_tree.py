import math
import operator
import numpy as np

class SegmentTree:
    def __init__(self, capacity, operation, identity_element):
        # Capacity is number of leaf nodes.  There are 2*capacity - 1 elements in the entire tree
        #   Nodes in tree: 2^0 + 2^1 + 2^2 + ... 2^(n - 1) = 2^n - 1
        #   If number of leaf nodes = capacity = 2^(n - 1)
        #   then there are 2*capacity - 1 = 2*2^(n - 1) - 1 = 2^n - 1 elements in entire tree
        #      Could also use height argument to get 2^(log_2(capacity)) - 1
        
        next_power_of_two = 2**math.ceil(math.log2(capacity))  # really the number of leaves.. perhaps change variable name
        self._capacity = next_power_of_two
        self._values = [identity_element for _ in range(2 * self._capacity - 1)]
        self._operation = operation
        self._identity_element = identity_element

    @property
    def capacity(self):
        return self._capacity

    def _reduce_helper(self, query_start, query_end, node, node_start, node_end):  # node_start/end may better be called node_min_index or similar
        if (query_start <= node_start and query_end >= node_end):
            return self._values[node]
        elif (node_end < query_start or node_start > query_end):
            return self._identity_element

        mid = node_start + (node_end - node_start) // 2
        return self._operation(self._reduce_helper(query_start, query_end, 2*node + 1, node_start, mid),
                               self._reduce_helper(query_start, query_end, 2*node + 2, mid + 1, node_end))

    def reduce(self, start=0, end=None):
    # TODO: Perhaps allow negative index like in openai implementation
        if end == None:
            end = self._capacity - 1
    
        if (start < 0 or end > self._capacity - 1 or start > end):
            raise RuntimeError("Invalid inputs")
        
        return self._reduce_helper(start, end, 0, 0, self._capacity - 1)

    def __setitem__(self, key, value):
        #TODO: ADD BOUNDS
        leaf = self._capacity - 1 + key
        self._values[leaf] = value
        
        parent = (leaf - 1) // 2
        while True:
            self._values[parent] = self._operation(self._values[2*parent + 1], self._values[2*parent + 2])

            if parent == 0:
                break
            parent = (parent - 1) // 2

    def __getitem__(self, key):
        if key < 0 or key > self._capacity - 1:
            raise RuntimeError("Index out of range")
            
        return self._values[self._capacity - 1 + key]


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super().__init__(capacity=capacity,
                         operation=operator.add,
                         identity_element=0.0)

    # This implementation (for sums) should be a bit faster than the general implementation
    def __setitem__(self, key, value):
        # TODO: ADD BOUNDS
        leaf = self._capacity - 1 + key
        diff = value - self._values[leaf]
        self._values[leaf] = value

        parent = (leaf - 1) // 2
        while True:
            self._values[parent] += diff
            if parent == 0:
                break
            parent = (parent - 1) // 2

    # TODO: There might be a way to do this a bit faster for a batch of indexes (rather than just call this repeatedly).  Probably requires a sorted list of prefix sums and sequentially building off previous explorations
    def prefix_sum_index(self, prefix_sum):
        # TODO: bounds check.. openai uses an epsilon which may be important
        index = 0
        while index < self._capacity - 1:
            if self._values[2*index + 1] > prefix_sum:
                index = 2*index + 1
            else:
                prefix_sum -= self._values[2*index + 1]
                index = 2*index + 2
        return index - self._capacity + 1  # same as index - (self._capacity - 1)

    def sum(self, start=0, end=None):
        return super().reduce(start, end)

        
class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super().__init__(capacity=capacity,
                         operation=min,
                         identity_element=float('inf'))

    def min(self, start=0, end=None):
        return super().reduce(start, end)


##sum_tree = SumSegmentTree(3)
##sum_tree[0] = 5
##sum_tree[1] = 7
##sum_tree[2] = 3
##sum_tree[3] = 9
##print(sum_tree._values)
##print(sum_tree.reduce(1, 2))
##print(sum_tree.prefix_sum_index(25))
##print(sum_tree[0], sum_tree[1], sum_tree[2], sum_tree[3])
            
##sum_tree = SegmentTree(capacity=3, operation=operator.add, identity_element=0.0)
##sum_tree[0] = 5
##sum_tree[1] = 7
##sum_tree[2] = 3
##sum_tree[3] = 9
##print(sum_tree._values)
##print(sum_tree.reduce(1, 2))
##print(sum_tree[0], sum_tree[1], sum_tree[2], sum_tree[3])

##min_tree = MinSegmentTree(3)
##min_tree[0] = 5
##min_tree[1] = 7
##min_tree[2] = 3
##min_tree[3] = 9
##print(min_tree._values)
##print(min_tree.reduce(1, 2))
##print(min_tree[0], min_tree[1], min_tree[2], min_tree[3])
