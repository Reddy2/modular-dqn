# Categorical Algorithm (C51)

import gym
import dqn.networks as nn
from dqn.agent import DQNAgent
import dqn.annealing_schedules
import dqn.algorithms
import dqn.experience_replay
import tensorflow.contrib.layers as layers

env = gym.make('CartPole-v1')
num_steps=100000

# Here, using C51, we demonstrate the Rainbow algorithm
# (C51, proportional prioritized replay, noisy net exploration, dueling nets, double-q, n-step learning) 
q_func = nn.DistributionalQNetwork([64], env.action_space.n, n=51, noisy_net=True, dueling=[32])

# Action selection in Rainbow is done using noisy nets with no epsilon
epsilon_scheduler = dqn.annealing_schedules.Constant(0)
action_selection = dqn.algorithms.EpsilonGreedy(epsilon_scheduler)

loss = dqn.algorithms.CategoricalAlgorithm(double_q=True)
update_target = dqn.algorithms.HardUpdate()

alpha_scheduler = dqn.annealing_schedules.Constant(0.7)
beta_scheduler = dqn.annealing_schedules.Constant(0.5)
memory = dqn.experience_replay.Proportional(capacity=100000, alpha_scheduler=alpha_scheduler, beta_scheduler=beta_scheduler)

agent = DQNAgent(network=q_func,
                 observation_space=env.observation_space,
                 action_space=env.action_space,
                 action_selection=action_selection,
                 loss=loss,
                 update_target=update_target,
                 memory=memory,
                 n_step=3,
                 update_target_network_frequency=200)

agent.load('save_test/rainbow')
agent.run(env, num_timesteps=num_steps, render=True)
