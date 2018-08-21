# Quantile Regression DQN (QR-DQN)

import gym
import dqn.networks as nn
from dqn.agent import DQNAgent
import dqn.annealing_schedules
import dqn.algorithms
import dqn.experience_replay
import tensorflow.contrib.layers as layers

env = gym.make('CartPole-v1')
num_steps=200000

# Here we combine the same improvements from Rainbow, but use QR instead of C51
# Note that we are still using a DistributionalQNetwork, but this network uses n as the number of quantiles rather than the number of atoms
# TODO: Do we want to allow noisy_net=False ? Does this make sense or not ?
q_func = nn.DistributionalQNetwork([64], env.action_space.n, n=75, noisy_net=True, dueling=[32])
epsilon_scheduler = dqn.annealing_schedules.Constant(0)
action_selection = dqn.algorithms.EpsilonGreedy(epsilon_scheduler)
loss = dqn.algorithms.QuantileRegressionLoss()
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
                 update_target_network_frequency=100)

agent.load('save/qr_dqn')
agent.run(env, num_timesteps=num_steps, render=True)
