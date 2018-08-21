# This file tests the original DQN algorithm and shows some ways of using the module

import gym
import dqn.networks as nn
from dqn.agent import DQNAgent
import dqn.annealing_schedules
import dqn.algorithms
import dqn.experience_replay
import tensorflow.contrib.layers as layers
import tensorflow as tf

# A custom Q-Network is made similarly to a PyTorch network
class Network(nn.QNetwork):
    # Note: __init__ must be overwritten
    def __init__(self):
        pass
    
    def forward(self, state):
        out = state
        out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.tanh)
        out = layers.fully_connected(out, num_outputs=env.action_space.n, activation_fn=None)
        return out

env = gym.make('CartPole-v1')
num_steps=100000

# We can use our custom network or use a network creation function (note support for noisy nets/dueling nets)
##q_func = nn.QNetwork(hiddens=[32], num_actions=env.action_space.n, noisy_net=True, dueling=[32])
q_func = Network()

# Note: CartPole learns extremely fast with NO exploration (use below line of code instead of the one after it) on good initializations
#       (Restarting a few times you may be able to get a reward of 500 in under 100 episodes, sometimes even half of that)
##epsilon_scheduler = dqn.annealing_schedules.Constant(0)
epsilon_scheduler = dqn.annealing_schedules.Linear(start=1, end=0.02, num_steps=7/8 * num_steps)
action_selection = dqn.algorithms.EpsilonGreedy(epsilon_scheduler)

# We use the standard DQN loss, which supports double-q learning.
# We can also change the network optimizer here (None will select the tf.train.AdamOptimizer() with it's default arguments, this argument may be moved to the DQNAgent itself)
loss = dqn.algorithms.TDErrorLoss(double_q=True, optimizer=None)
update_target = dqn.algorithms.HardUpdate()

# We can use proportional or rank-based prioritized replay (proportional seems to be prefered by many papers)
# Simple, non-prioritized replay is also implemented

alpha_scheduler = dqn.annealing_schedules.Constant(0.7)
beta_scheduler = dqn.annealing_schedules.Constant(0.5)
memory = dqn.experience_replay.Proportional(capacity=50000, alpha_scheduler=alpha_scheduler, beta_scheduler=beta_scheduler)
##memory = dqn.experience_replay.RankBased(capacity=50000, alpha_scheduler=alpha_scheduler, beta_scheduler=beta_scheduler)
##memory = dqn.experience_replay.Simple(capacity=50000)

# Below we add n-step learning with the parameter n-step
# Not yet supported: Frame skipping will be added in the future
agent = DQNAgent(network=q_func,
                 observation_space=env.observation_space,
                 action_space=env.action_space,
                 action_selection=action_selection,
                 loss=loss,
                 update_target=update_target,
                 memory=memory,
                 n_step=3,
                 update_target_network_frequency=2000)

agent.train(env, num_timesteps=num_steps, render=False)

# We can save and load an agent
# Note: Currently this only saves the weights of the network -- the entire agent must be recreated (or reused, as would happen here) before calling load
##agent.save('/tmp/save_test/test')
##agent.load('/tmp/save_test/test')
