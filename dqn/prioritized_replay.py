import numpy as np
import math
import random
from collections import namedtuple

Experience = namedtuple("Experience", ["state", "reward", "action", "next_state", "terminal"])

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

    def peak(self):
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
    

class RankBased:
    # TODO: Precompute segments (requires a known batch_size = num_segments)
    def __init__(self, capacity, num_insertions_until_sort=float('inf')):
        self._priority_queue = MaxPriorityQueue(capacity)
        self.num_insertions_until_sort = num_insertions_until_sort
        self._insertions_since_sort = 0
        #self._batch_size = batch_size

    def _experience_probs(self, alpha):
        # Returns probability of each experience in the priority queue by index (ordered) <-- make this clearer
        ranks = np.arange(1, len(self._priority_queue) + 1)
        priorities = 1 / ranks
        powers = np.power(priorities, alpha)
        probabilities = powers / np.sum(powers)

        return probabilities

    def _segment(self, probs, num_segments):
        ## TODO: WRITE DOC-STRING THAT EXPLAINS THE EXTRA SEGMENT AT THE END (and says N + 1 numbers)
        ## TODO: Talk about how this algorithm isn't perfect:  Note the addition of cdf part (either way) makes it strange
        cdf = 0
        prob_per_segment = 1 / num_segments
        next_prob_boundary = prob_per_segment
        segment_starts = [0]
        
        for i in range(len(probs)):
            if cdf >= next_prob_boundary:
                segment_starts.append(i)
                next_prob_boundary += prob_per_segment

            cdf += probs[i]

        segment_starts.append(len(self._priority_queue))
        return segment_starts           

    # TODO: Perhaps allow a variable number of variables to be stored
    # TODO: We are now storing gamma rather than terminal, also state_tpn rather than state_tp1 (next_state), etc..  Update variable names
    def store(self, state, action, reward, next_state, terminal):
        experience = Experience(state, action, reward, next_state, terminal)
        max_priority = self._priority_queue.max_priority()
        self._priority_queue.insert(max_priority, experience)

        if self._insertions_since_sort >= self.num_insertions_until_sort:
            self.sort()
            self._insertions_since_sort = 0

    def update_priorities(self, ids, priorities):
        self._priority_queue.update_priorities(ids, priorities)
        
    def sample(self, batch_size, alpha, beta):
        ### TODO: WRITE DOC-STRING
        ### TODO: ERROR WHEN SAMPLING WITHOUT ENOUGH MEMORIES IN STORAGE
        experiences = []
        order_ids = []
        sampled_probs = np.zeros(batch_size)

        all_probs = self._experience_probs(alpha)
        segments = self._segment(all_probs, batch_size)

        # Sample one transition from each segment (with each segment being of nearly equal probability)
        for i in range(len(segments) - 1):
            # Sample uniformly within each segment
            index = random.randint(segments[i], segments[i + 1] - 1)  # sample in range [start, next_start)
            _, order_id, experience = self._priority_queue[index]
            
            experiences.append(experience)
            order_ids.append(order_id)
            sampled_probs[i] = all_probs[index]

        min_prob = all_probs[-1]
        max_weight = (len(self._priority_queue) * min_prob)**(-beta)
        weights = np.power(len(self._priority_queue) * sampled_probs, -beta) / max_weight

        states, actions, rewards, next_states, is_terminals = zip(*experiences) 
        return states, actions, rewards, next_states, is_terminals, weights, order_ids

    def sort(self):
        self._priority_queue.sort()

    def __len__(self):
        return len(self._priority_queue)




###### The below (and remaining text) are notes on the algorithm implementation and some commented out tests #####

## Notes on algorithm implementation:
## Two paragraphs from the paper: https://arxiv.org/pdf/1511.05952.pdf TODO: Put in name of paper

## For the rank-based variant, we can approximate the cumulative density function with a piecewise
## linear function with k segments of equal probability. The segment boundaries can be precomputed
## (they change only when N or α change). At runtime, we sample a segment, and then sample uniformly
## among the transitions within it. This works particularly well in conjunction with a minibatchbased
## learning algorithm: choose k to be the size of the minibatch, and sample exactly one transition
## from each segment – this is a form of stratified sampling that has the added advantage of balancing
## out the minibatch (there will always be exactly one transition with high magnitude δ, one with
## medium magnitude, etc).

## Our final
## solution was to store transitions in a priority queue implemented with an array-based binary heap.
## The heap array was then directly used as an approximation of a sorted array, which is infrequently
## sorted once every 10^6
## steps to prevent the heap becoming too unbalanced. This is an unconventional
## use of a binary heap, however our tests on smaller environments showed learning was unaffected
## compared to using a perfectly sorted array. This is likely due to the last-seen TD-error only being a
## proxy for the usefulness of a transition and our use of stochastic prioritized sampling. A small improvement
## in running time came from avoiding excessive recalculation of partitions for the sampling
## distribution. We reused the same partition for values of N that are close together and by updating
## α and β infrequently. Our final implementation for rank-based prioritization produced an additional
## 2%-4% increase in running time and negligible additional memory usage. This could be reduced
## further in a number of ways, e.g. with a more efficient heap implementation, but it was good enough
## for our experiments.

# Understanding the first paragraph: TODO: Perhaps put this right under the first paragraph

## The CDF can be used with a uniform distribution for sampling from a discrete "pdf"
# For example: X can be {1, 2, 3}.  P(X = x) = [0.1, 0.4, 0.5]  (i.e. P(X = 2) = 0.4)
# We sample from a uniform distribution u = U[0, 1]
# We then use the CDF and u to sample from our discrete distribution:
# if u < 0.1: return 1
# if u < 0.1 + 0.4: return 2
# if u < 0.1 + 0.4 + 0.5: return 3
# Note: The CDF is P(X < x) =  [0.1, 0.1 + 0.4, 0.1 + 0.4 + 0.5] = [0.1, 0.5, 1]

# "we can approximate the cumulative density function with a piecewise
# linear function with k segments of equal probability"

# seems to mean the following:
# Take k segments, so each has equal probability 1 / k
# Let's say k = 3, so we get 1/3
# Then the cdf is [1/3, 2/3, 3/3]
# So there is a (horizontal) line with value 1/3, which jumps to 2/3, and then jumps to 1

# So we build a priority queue and put the priorities in there
# We sample from the priority queue using the CDF technique above
# Every once in a while we sort the priority queue


# Notes on rank-based implementation:
# The maximum weight will be the weight with the smallest probability
# The lowest probability will be the experience with the smallest priority
# The smallest priority will be 1 / max_rank = 1 / N

# Since only P(i) changes, the maximum weight is the one with the smallest P(i), min_i(P(i))
# w_i = (N * P(i))^(-beta)
# w_j = (N * P(j))^(-beta) / max_i(w_i)   # TODO: This is recursive and makes no sense !!!!!
# i is over the weights we have seen so far, N is the size of the priority queue's current size (NOT its capacity)
# TODO: If we precompute then N may be the size of the precomputed "N" !!

# min_prob = probabilities[-1] # Since the probabilties are done by 1 / rank(i) => min = 1 / rank(N)
        

## Tests 

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


##memory = RankedExperienceMemory(memory_size=8, batch_size=2)
##memory.store(1, 2, 3, 4, False)
##memory.store(6, 7, 8, 9, True)
##memory.store(10, 11, 12, 13, False)
##memory.store(14, 15, 16, 17, False)
##memory.store(18, 19, 20, 21, False)

#print(memory.sample())


##def experience_probs(n, alpha):
##    # Returns probability of each experience in the priority queue by index (ordered) <-- make this clearer
##    ranks = np.arange(1, n + 1)
##    priorities = 1 / ranks
##    powers = np.power(priorities, alpha)
##    probabilities = powers / np.sum(powers)
##
##    return probabilities
##
##def segment(n, probs, num_segments):
##    ## TODO: WRITE DOC-STRING THAT EXPLAINS THE EXTRA SEGMENT AT THE END (and says returns N+1 numbers)
##    cdf = 0
##    prob_per_segment = 1 / num_segments
##    next_prob_boundary = prob_per_segment
##    segment_starts = [0]
##    
##    for i in range(len(probs)):
##        if cdf >= next_prob_boundary:
##            segment_starts.append(i)
##            next_prob_boundary += prob_per_segment
##
##        cdf += probs[i]
##
##    segment_starts.append(n)
##    # perhaps add the end of the beginning of the fake next segment, or use tuples with (begin, end)
##
##    return segment_starts
##
##
##def print_segments(probs, segments):
##    for i in range(len(segments) - 1):
##        print(np.sum(probs[segments[i]: segments[i + 1]]))
##
##n = 10
##probs = experience_probs(n, 0.5)
##cdf = np.cumsum(probs)
##segments = segment(n, probs, 5)
##print(probs, segments)
##print_segments(probs, segments)   # For large n we get segments of nearly equal probability

##num_samples = 50
##experiences = [Experience(*exper) for exper in np.random.randint(0, 5, size=(num_samples, 5))]
##memory = RankedExperienceMemory(50)
##
##for experience in experiences:
##    memory.store(*experience)
##
##batch_size = 32
##mems, weights, ids = memory.sample(batch_size, 0.5, 0.5)
##memory.update_priorities(ids, np.random.rand(batch_size))
##print(memory._priority_queue._heap)
##print(memory._priority_queue)
##
##memory.sort()
##print(memory._priority_queue._heap)
##print(memory._priority_queue)

    

##print(memory._priority_queue._heap)
##print()
##print(memory._priority_queue)
