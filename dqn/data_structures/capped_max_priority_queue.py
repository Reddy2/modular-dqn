import math  # TODO: We can probably replace math.floor((index - 1) / 2) with (index - 1) // 2

## Max priority queue with a maximum capacity.  Removes oldest elements first.  No pop() currently supported
## TODO: perhaps rename this to something to indicate the cap (CappedMaxPriorityQueue)
class MaxPriorityQueue:
    def __init__(self, capacity):
        self._heap = []
        self._capacity = capacity
        self._oldest_order_index = 0
        self._order_to_priority = {}

    def _swap_indexes(self, parent_priority_index, child_priority_index):
        # TODO: Maybe make comment about how this should be called before swapping heap elements (order_ids)
        parent_order_index = self._heap[parent_priority_index][1]
        child_order_index = self._heap[child_priority_index][1]

        self._order_to_priority[parent_order_index] = child_priority_index
        self._order_to_priority[child_order_index] = parent_priority_index
        
    def _up_heap(self, index):
        if index > 0:
            parent_index = math.floor((index - 1) / 2)

            if self._heap[parent_index] < self._heap[index]:
                # Swap parent with child
                self._swap_indexes(parent_index, index)
                self._heap[parent_index], self._heap[index] = self._heap[index], self._heap[parent_index]
                
                self._up_heap(parent_index)
            
    def _down_heap(self, index):
        left_index, right_index = 2*index + 1, 2*index + 2
        largest = index

        if left_index < len(self._heap) and self._heap[left_index] > self._heap[largest]:
            largest = left_index
        if right_index < len(self._heap) and self._heap[right_index] > self._heap[largest]:
            largest = right_index

        if largest != index:
            # Swap parent with child
            self._swap_indexes(index, largest)
            self._heap[index], self._heap[largest] = self._heap[largest], self._heap[index]
            
            self._down_heap(largest)

    def insert(self, priority, data):
        if len(self._heap) < self._capacity:
            self._heap.append((priority, self._oldest_order_index, data))
            
            priority_index = len(self._heap) - 1
            self._order_to_priority[self._oldest_order_index] = priority_index

            self._up_heap(priority_index)
        else:
            oldest_priority_index = self._order_to_priority[self._oldest_order_index]
            self._heap[oldest_priority_index] = (priority, self._oldest_order_index, data)
            
            self._up_heap(oldest_priority_index)
            self._down_heap(oldest_priority_index)

        self._oldest_order_index = (self._oldest_order_index + 1) % self._capacity

    def update_priorities(self, order_indexes, priorities):
        for order_index, priority in zip(order_indexes, priorities):
            priority_index = self._order_to_priority[order_index]
            data = self._heap[priority_index][2]
            self._heap[priority_index] = (priority, order_index, data)
            
            self._up_heap(priority_index)
            self._down_heap(priority_index)

    # Pop seems to make the structure much more complicated (this may not be true), and in our case it is unneeded
    # For example, if we pop an element and add a new one in, there is a (very high: 1 - 1/n) chance that
    #    there will be two elements in the heap with the same order_id, which would make our dictionary approach
    #    implausible (two identical keys)

    # There may be a shifting technique, but this will probably require shifting upto n elements in the array and
    #    this technique may have its own issues (meaning perhaps there isn't a shifting technique)

##    def pop(self):
##        max_priority = self._heap[0]
##        last_item = self._heap.pop(-1)
##
##        if len(self._heap) > 1:
##            # Remove the last value of the heap and put it as the root, then bring it down the heap
##            ### TODO: Do we need to look after other stuff here (yes) ?!
##            self._heap[0] = last_item
##            self._down_heap(0)
##
##        return max_priority

    def peek(self):
        return self._heap[0]

    def max_priority(self):
        if len(self._heap) == 0:    # TODO: Should we use a value other than 0 (perhaps allow init to do this)?!
            return 0
        
        return self._heap[0][0]

    # TODO: Perhaps implement slicing (it may already be implemented if we return the whole tuple!)
    # https://stackoverflow.com/questions/2936863/python-implementing-slicing-in-getitem
    def __getitem__(self, key):
        """Returns (priority, order_id, data)"""
        return self._heap[key]

    def sort(self):
        sorted_heap = sorted(self._heap, reverse=True)
        
        for priority_index in range(len(sorted_heap)):
            order_index = sorted_heap[priority_index][1]
            self._order_to_priority[order_index] = priority_index

        self._heap = sorted_heap

    def _print_tree(self, string="", index=0, indent=0):
        string += '\t' * indent + str(self._heap[index]) + '\n'
        left_child_index, right_child_index = 2*index + 1, 2*index + 2

        if left_child_index < len(self._heap):
            string = self._print_tree(string, left_child_index, indent + 1)
        if right_child_index < len(self._heap):
            string = self._print_tree(string, right_child_index, indent + 1)

        # Remove trailing '\n' from last element in the heap (len('\n') == 1)
        if index == 0:
            return string[:-1]
        
        return string

    def __str__(self):
        return self._print_tree()

    def __len__(self):
        return len(self._heap)


##test = MaxPriorityQueue(capacity=6)
##test.insert(5, '0th')
##test.insert(2, '1st')
##test.insert(4, '2nd')
##test.insert(7, '3rd')
##test.insert(1, '4th')
##test.insert(6, '5th')
##test.insert(8, '6th')
##test.insert(3, '7th')
##test.insert(0, '8th')
##print(test)
##print(test._heap)
##print("Order to priority: ", test._order_to_priority)
##
##test.sort()
##print(test)
##print(test._heap)
##print("Order to priority: ", test._order_to_priority)
##
##test.update_priorities([0, 1, 2, 3], [5, 8, 3, 0])
##print(test)
##print(test._heap)
##print("Order to priority: ", test._order_to_priority)
##
##print(test[0], test[1], test[5])
