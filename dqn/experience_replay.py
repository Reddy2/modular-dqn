import numpy as np
import random
from collections import namedtuple, deque
from dqn.data_structures import SumSegmentTree, MinSegmentTree, MaxPriorityQueue

Experience = namedtuple("Experience", ["state_t", "action_t", "reward_tn", "state_tpn", "gamma_n"])

class Simple:
    def __init__(self, capacity):
        self._memories = deque(maxlen=capacity)

    def store(self, state_t, action_t, reward_tn, state_tpn, gamma_n):
        experience = Experience(state_t, action_t, reward_tn, state_tpn, gamma_n)
        self._memories.append(experience)
        
    def sample(self, batch_size):
        # TODO: Perhaps ensure len(self._memories) >= batch_size
        # Note: random.sample does not allow repeats.  Do we want to allow them ?
        return random.sample(self._memories, batch_size)

    def __len__(self):
        return len(self._memories)
    

# TODO: Think about storing the data directly in the segment trees, similar to how we do it for rank based
class Proportional:
    def __init__(self, capacity, alpha_scheduler, beta_scheduler, epsilon=1e-5):
        self._capacity = capacity  # NOTE: The capacity here might be different than the segment trees' (the next power of 2).  Does this cause any issues ? (I don't believe so)
        self._alpha_scheduler = alpha_scheduler
        self._beta_scheduler = beta_scheduler
        self._sum_tree = SumSegmentTree(capacity)
        self._min_tree = MinSegmentTree(capacity)    # Is it more efficient to use a Min Priority Queue with a cap similar to the one in RankBased ?
        self._epsilon = epsilon
        self._memories = []
        self._oldest_index = 0
        self._max_priority = 1

    def store(self, state_t, action_t, reward_tn, state_tpn, gamma_n):
        experience = Experience(state_t, action_t, reward_tn, state_tpn, gamma_n)
        
        if len(self._memories) < self._capacity:
            self._memories.append(experience)
        else:
            self._memories[self._oldest_index] = experience

        self._sum_tree[self._oldest_index] = self._max_priority
        self._min_tree[self._oldest_index] = self._max_priority
        self._oldest_index = (self._oldest_index + 1) % self._capacity

    def sample(self, t, batch_size):
        if len(self._memories) < batch_size:
            raise RuntimeError("Not enough stored memories (" + str(len(self._memories)) + ") for batch_size of size " + str(batch_size))
        
        total_priority_sum = self._sum_tree.sum()
        segment_indexes = np.linspace(self._epsilon, total_priority_sum, batch_size + 1)  # The smallest possible priority is of size |0| + eps = eps
        
        indexes = []
        for i in range(batch_size):
            prefix_sum = np.random.uniform(low=segment_indexes[i], high=segment_indexes[i + 1])
            indexes.append(self._sum_tree.prefix_sum_index(prefix_sum))

        total_sum = self._sum_tree.sum()        
        sampled_probs = np.zeros(batch_size)
        experiences = []
        for i, index in enumerate(indexes):
            prob = self._sum_tree[index] / total_sum
            sampled_probs[i] = prob
            experiences.append(self._memories[index])
            
        min_prob = self._min_tree.min() / total_sum
        beta = self._beta_scheduler.value(t)
        max_weight = np.power(len(self._memories) * min_prob, -beta)
        weights = np.power(len(self._memories) * sampled_probs, -beta) / max_weight

        states_t, actions_t, rewards_tn, stats_tpn, gammas_n = zip(*experiences)
        return states_t, actions_t, rewards_tn, stats_tpn, gammas_n, weights, indexes

    def update_priorities(self, t, indexes, priorities):
        alpha = self._alpha_scheduler.value(t)
        priorities = np.abs(priorities) + self._epsilon  # Note: Our implementation may not really be effected by removing an epsilon (uniform sampling from bounds.. unless a priority of 0 is on the end bounds and so prefix_sum never goes that far)
        
        for index, priority in zip(indexes, priorities):
            self._sum_tree[index] = priority**alpha
            self._min_tree[index] = priority**alpha
            self._max_priority = max(priority**alpha, self._max_priority)

    def __len__(self):
        return len(self._memories)


class RankBased:
    # TODO: Precompute segments (requires a known batch_size = num_segments)
    #       Note the segments change based on N AND alpha
    def __init__(self, capacity, alpha_scheduler, beta_scheduler, epsilon=1e-5, num_stores_until_sort=float('inf')):
        self._alpha_scheduler = alpha_scheduler
        self._beta_scheduler = beta_scheduler
        self._priority_queue = MaxPriorityQueue(capacity)
        self.num_stores_until_sort = num_stores_until_sort
        self._stores_since_sort = 0

    def _experience_probs(self, alpha):
        # Returns probability of each experience in the priority queue by index (ordered) <-- make this clearer
        ranks = np.arange(1, len(self._priority_queue) + 1)
        priorities = 1 / ranks
        powers = np.power(priorities, alpha)
        probabilities = powers / np.sum(powers)

        return probabilities

    def _segment(self, probs, num_segments):
        ## TODO: Explain extra segment at end in doc-string (and says N + 1 numbers)
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

    def store(self, state_t, action_t, reward_tn, state_tpn, gamma_n):
        experience = Experience(state_t, action_t, reward_tn, state_tpn, gamma_n)
        max_priority = self._priority_queue.max_priority()
        self._priority_queue.insert(max_priority, experience)

        if self._stores_since_sort >= self.num_stores_until_sort:
            self.sort()
            self._stores_since_sort = 0
        else:
            self._stores_since_sort += 1
        
    def sample(self, t, batch_size):
        ### TODO: Error when sampling without enough memories in storage
        experiences = []
        order_ids = []
        sampled_probs = np.zeros(batch_size)

        alpha = self._alpha_scheduler.value(t)
        all_probs = self._experience_probs(alpha)
        prob_segments = self._segment(all_probs, batch_size)

        # Sample one transition from each segment (with each segment being of nearly equal probability)
        for i in range(len(prob_segments) - 1):
            index = random.randint(prob_segments[i], prob_segments[i + 1] - 1)  # sample in range [start, next_start)
            _, order_id, experience = self._priority_queue[index]
            
            experiences.append(experience)
            order_ids.append(order_id)
            sampled_probs[i] = all_probs[index]

        min_prob = all_probs[-1]   # Note: This should eventually become a constant.. might be a faster method
        beta = self._beta_scheduler.value(t)
        max_weight = (len(self._priority_queue) * min_prob)**(-beta)
        weights = np.power(len(self._priority_queue) * sampled_probs, -beta) / max_weight

        states_t, actions_t, rewards_tn, stats_tpn, gammas_n = zip(*experiences) 
        return states_t, actions_t, rewards_tn, stats_tpn, gammas_n, weights, order_ids

    def update_priorities(self, t, indexes, priorities):
        priorities = np.abs(priorities)
        self._priority_queue.update_priorities(indexes, priorities)

    def sort(self):
        self._priority_queue.sort()

    def __len__(self):
        return len(self._priority_queue)



##if __name__ == '__main__':
##    test = Standard(capacity=5)
##    test.store(1, 2, 3, 4, 0)
##    test.store(4, 5, 6, 7, 0)
##    test.store(8, 9, 10, 11, 0)
##    test.store(12, 13, 14, 15, 0)
##    test.store(16, 17, 18, 19, 0)
##    print(test._memories)
##    print(test.sample(3))


##if __name__ == '__main__':
##    import annealing_schedules
##    from data_structures import SumSegmentTree, MinSegmentTree, MaxPriorityQueue
##    
##    alpha_scheduler = annealing_schedules.Constant(0.7)
##    beta_scheduler = annealing_schedules.Constant(0.5)
##    test = RankBased(8, alpha_scheduler, beta_scheduler)
####    test = Proportional(8, alpha_scheduler, beta_scheduler)
##    test.store(1, 2, 3, 4, 0)
##    test.store(4, 5, 6, 7, 0)
##    test.store(8, 9, 10, 11, 0)
##    samples = test.sample(0, 3)
##    test.update_priorities(2, samples[-1], [0.67, 1.23, 0.23])
##    print(test.sample(1, 3))
##    #print(test._max_priority)





###### The below (and remaining text) are notes on the rank based algorithm implementation and some commented out tests #####

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
