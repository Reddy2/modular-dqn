# An interesting solution DQN found is shown here
# Some solutions converge to the pole standing almost perfectly upright in the center of the screen

import gym
import dqn.networks as nn
from dqn.agent import DQNAgent
import dqn.annealing_schedules
import dqn.algorithms
import dqn.experience_replay
import tensorflow.contrib.layers as layers
import tensorflow as tf

class Network(nn.QNetwork):
    def __init__(self):
        pass
    
    def forward(self, state):
        out = state
        out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.tanh)
        out = layers.fully_connected(out, num_outputs=env.action_space.n, activation_fn=None)
        return out

env = gym.make('CartPole-v1')

num_steps=100000
q_func = Network()
epsilon_scheduler = dqn.annealing_schedules.Constant(0)
action_selection = dqn.algorithms.EpsilonGreedy(epsilon_scheduler)
loss = dqn.algorithms.TDErrorLoss(double_q=True, optimizer=None)
update_target = dqn.algorithms.HardUpdate()
alpha_scheduler = dqn.annealing_schedules.Constant(0.7)
beta_scheduler = dqn.annealing_schedules.Constant(0.5)
memory = dqn.experience_replay.Proportional(capacity=50000, alpha_scheduler=alpha_scheduler, beta_scheduler=beta_scheduler)

agent = DQNAgent(network=q_func,
                 observation_space=env.observation_space,
                 action_space=env.action_space,
                 action_selection=action_selection,
                 loss=loss,
                 update_target=update_target,
                 memory=memory,
                 n_step=3,
                 update_target_network_frequency=2000)

agent.load('data/test')
agent.run(env, num_timesteps=num_steps, render=True)
