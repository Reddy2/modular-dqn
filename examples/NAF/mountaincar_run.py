# NAF Algorithm (continuous action space)

import gym
import tensorflow as tf
import dqn.networks as nn
from dqn.agent import DQNAgent
import dqn.annealing_schedules
import dqn.algorithms
import dqn.experience_replay

env = gym.make('MountainCarContinuous-v0')
num_steps=150000

q_func = nn.NAFQNetwork(shared_hiddens=[100, 100], mu_hiddens=[], value_hiddens=[], l_hiddens=[], action_space_shape=env.action_space.shape, activation_fn=tf.tanh, diagonal_l=False, noisy_net=False)
stddev_scheduler = dqn.annealing_schedules.Linear(start=0, end=0, num_steps=num_steps)
action_selection = dqn.algorithms.GaussianRandomProcess(stddev_scheduler)
loss = dqn.algorithms.NAFLoss()  #TODO: ADD IN ALL OPTIONS HERE AND IN OTHER ONES
update_target = dqn.algorithms.SoftUpdate(tau=0.001)
alpha_scheduler = dqn.annealing_schedules.Constant(0.7)
beta_scheduler = dqn.annealing_schedules.Constant(0.5)
memory = dqn.experience_replay.Proportional(capacity=1000000, alpha_scheduler=alpha_scheduler, beta_scheduler=beta_scheduler)

agent = DQNAgent(network=q_func,
                 observation_space=env.observation_space,
                 action_space=env.action_space,
                 action_selection=action_selection,
                 loss=loss,
                 update_target=update_target,
                 memory=memory,
                 n_step=1,
                 batch_size=100,
                 discount_factor=0.99,
                 replay_period=1,
                 replays_per_step=5,
                 update_with_replay=True,
                 update_target_network_frequency=1)

agent.load('data/naf')
agent.run(env, num_timesteps=num_steps, render=True)
