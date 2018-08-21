import dqn.experience_replay
import dqn.annealing_schedules
import dqn.algorithms
import dqn.runner
import numpy as np
import tensorflow as tf
import os

class DQNAgent:
    # TODO: Update default values
    def __init__(self,
                 network,
                 observation_space,
                 action_space,
                 action_selection,  # TODO: Make a default option
                 loss,              # TODO: Make a default option based on net type
                 update_target,     # TODO: Make a default option
                 memory,
                 batch_size=32,   # TODO: Probably make this a component variable for memories.. right now we have to put it in for sample but for RankBased it will need to be on __init__
                 n_step=1,
                 discount_factor=1.0,
                 replay_period=1,
                 replays_per_step=1,
                 update_with_replay=False,
                 update_target_network_frequency=500):

        self.observation_space = observation_space
        self.action_space = action_space
        self.batch_size = batch_size
        self.n_step = n_step
        self.discount_factor = discount_factor
        self.replay_period = replay_period
        self.replays_per_step = replays_per_step
        self.update_with_replay = update_with_replay
        self.update_target_network_frequency = update_target_network_frequency
        self.memory = memory

        self.num_acts = 0
        self.num_observations = 0

        # These are used by other components when those components are being built
        self.network_spec = network
        self.loss_spec = loss
        self.action_selection_spec = action_selection
        self.update_target_spec = update_target

        # TODO: Perhaps rename these to build_func to not confuse with the network building above (which aren't functions but part of a graph)
        #       Perhaps also allow the original build to allow the components to be built into the graph, like networks
        self.online_net = self.network_spec.build('online_net')
        self.target_net = self.network_spec.build('target_net')   # TODO: These variables can be set to non-trainable.. perhaps we should add this option

        self._loss = loss.build(self)
        self._act = action_selection.build(self)
        self._update_target = update_target.build(self)

        self._saver = tf.train.Saver()
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
        state_t, action_t, reward_tn, state_tpn, gamma_n, importance_weights, indexes = self.memory.sample(t=self.num_observations, batch_size=self.batch_size)
        priorities = self._loss(state_t, action_t, reward_tn, state_tpn, gamma_n, importance_weights)
        self.memory.update_priorities(t=self.num_observations, indexes=indexes, priorities=priorities)

    def observe(self, state_t, action_t, reward_tn, state_tpn, gamma_n):
        # TODO: The memory should have access to the agent's internals, similar to how other components are built
        self.memory.store(state_t, action_t, reward_tn, state_tpn, gamma_n)
        
        if self.num_observations % self.replay_period == 0 and len(self.memory) >= self.batch_size:
            for _ in range(self.replays_per_step):
                self._replay()
                if self.update_with_replay:
                    self._update_target()

        if not self.update_with_replay and self.num_observations % self.update_target_network_frequency == 0:
            self._update_target()

        self.num_observations += 1

    # TODO: Add a save_every option (for episodes or timesteps.. but might need to only use one because this might conflic with NStepRunner.. hmmm)
    def train(self, enviornment, num_episodes=None, num_timesteps=None, render=False):
        # TODO: Perhaps have some sort of reset option here to reset the agent to its defaults (although this would require some extra infrastructure or a total refactoring)
        runner = dqn.runner.NStepRunner(agent=self, enviornment=enviornment, n=self.n_step, discount_factor=self.discount_factor)
        runner.run(num_episodes, num_timesteps, training=True, render=render)

    # TODO: runner doesn't need to be an NStepRunner here since the code only uses the act function.. maybe use a different type of runner
    #       Might also want to just make this one function since exactly the same exact training=True
    def run(self, enviornment, num_episodes=None, num_timesteps=None, render=False):
        # TODO: Perhaps have some sort of reset option here to reset the agent to its defaults (although this would require some extra infrastructure or a total refactoring)
        runner = dqn.runner.NStepRunner(agent=self, enviornment=enviornment, n=self.n_step, discount_factor=self.discount_factor)
        runner.run(num_episodes, num_timesteps, training=False, render=render)

    # TODO:  This currently only saves the variables from the network
    #        Right now the user has to build the entire agent again (manually) before loading
    #        We can make this an automatic process by storing the relevant instance variables in a pickle with the tensorflow files (see openai baselines code)
    #        This is dangerous to do though.  Say we use a global step.  The global step will (I believe) be restored with the network
    #           but if the user wants to start somewhere new (say a new epsilon annealing value, assuming they are eventually implemented in tensorflow)
    #           then the global step will still be set.  We can't just reset the global step since this may affect something else we didn't want to restart
    #           This may mean every module needs it's own step count and a reset option
    #           This may be better suited as a separate load function (load_tf_model vs. load_agent or something)
    def save(self, fname):
        # TODO: Talk about file path it returns (for loading)
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        return self._saver.save(self._sess, fname)

    def load(self, fname):
        # TODO: Get either the old agent or all it's instance variables
        # Call init on this agent (or is it already inited?)
        # Either way, we need to rebuild all the graph
        # Then we can restore
        self._saver.restore(self._sess, fname)
