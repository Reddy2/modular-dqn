import dqn.prioritized_replay
import dqn.annealing_schedules
import dqn.backend
import dqn.runner
import numpy as np
import tensorflow as tf

# TODO: Add save/load
class DQNAgent:
    # TODO: Update default values
    def __init__(self,
                 network,
                 observation_space_shape,
                 num_actions,
                 action_selection,  # TODO: Make a default option
                 loss,              # TODO: Make a default option based on net type
                 batch_size=32,
                 n_step=3,
                 discount_factor=1.0,
                 memory_size=50000,
                 replay_period=1,
                 update_target_network_frequency=500,
                 alpha_scheduler=dqn.annealing_schedules.Constant(0.7),
                 beta_scheduler=dqn.annealing_schedules.Constant(0.5)):

        self.observation_space_shape = observation_space_shape
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.n_step = n_step
        self.discount_factor = discount_factor
        self.memory_size = memory_size
        self.replay_period = replay_period
        self.update_target_network_frequency = update_target_network_frequency
        
        # TODO: These should be options for the memory explicitly (not for the agent)
        self.alpha_scheduler = alpha_scheduler
        self.beta_scheduler = beta_scheduler
        self.memory = dqn.prioritized_replay.RankBased(memory_size, batch_size)

        self.num_acts = 0
        self.num_observations = 0

        # These are used by other components when those components are being built
        self.network_spec = network
        self.loss_spec = loss
        self.action_selection_spec = action_selection

        # TODO: Perhaps rename these to build_func to not confuse with the network building above (which aren't functions but part of a graph)
        #       Perhaps also allow the original build to allow the components to be built into the graph, like networks
        self.online_net = self.network_spec.build('online_net')
        self.target_net = self.network_spec.build('target_net')   # TODO: These variables can be set to non-trainable.. perhaps we should add this option

        self._loss = loss.build(self)
        self._act = action_selection.build(self)
        self._update_target = dqn.backend.UpdateTarget().build(self)

        self._sess = tf.Session()
        self._sess.__enter__()
        self._sess.run(tf.global_variables_initializer())

        # The weights in q_online/q_target will be randomly initialized, so we must instantly update q_target
        self._update_target()

    def act(self, state):
        action = self._act(state)
        self.num_acts += 1
        return action

    def _replay(self):
        alpha, beta = self.alpha_scheduler.value(self.num_observations), self.beta_scheduler.value(self.num_observations)
        state_t, action_t, reward_tn, state_tpn, gamma_n, importance_weights, indexes = self.memory.sample(self.batch_size, alpha, beta)
        priorities = self._loss(state_t, action_t, reward_tn, state_tpn, gamma_n, importance_weights)
        self.memory.update_priorities(indexes, np.abs(priorities))
        # TODO/Note: We don't need a minimum epsilon on line above since the probabilities are always nonzero, but perhaps we should implement one anyway so this code works for the other type of prioritized replay (check this is allowed for rank-based)

    def observe(self, state_t, action_t, reward_tn, state_tpn, gamma_n):
        self.memory.store(state_t, action_t, reward_tn, state_tpn, gamma_n)

        # TODO: Sort the replay memory for rank-based (This should probably be performed by the memory object itself every n insertions)
        #       The memory should have access to the agent's internals, similar to how other components are built
        
        if self.num_observations % self.replay_period == 0 and len(self.memory) >= self.batch_size:
            self._replay()

        if self.num_observations % self.update_target_network_frequency == 0:
            self._update_target()

        self.num_observations += 1

    def train(self, enviornment, num_episodes=None, num_timesteps=None, render=False):
        # TODO: Perhaps have some sort of reset option here to reset the agent to its defaults (although this would require some extra infrastructure or a total refactoring)
        runner = dqn.runner.NStepRunner(agent=self, enviornment=enviornment, n=self.n_step, discount_factor=self.discount_factor)
        runner.run(num_episodes, num_timesteps, render)
