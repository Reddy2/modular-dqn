import gym
import dqn.networks as nn
from dqn.agent import DQNAgent
import dqn.annealing_schedules
import dqn.backend
import tensorflow as tf
import tensorflow.contrib.layers as layers


env = gym.make('CartPole-v1')

class Network(nn.QNetwork):
    def __init__(self):
        pass
    
    def forward(self, state):
        out = state
        out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=env.action_space.n, activation_fn=None)
        return out

# CartPole learns extremely fast with NO exploration on good initializations (restarting a few times you may be able to get a reward of 500 in under 100 episodes, sometimes even half of that)


# Simple Q-Network.  Original Algorithm 
##q_func = nn.QNetwork([32], env.action_space.n, noisy_net=True, dueling=[32])
###q_func = nn.QNetwork([64], env.action_space.n)
##q_func = Network()
###epsilon_scheduler = dqn.annealing_schedules.Linear(start=1, end=0.02, num_steps=10000)
##epsilon_scheduler = dqn.annealing_schedules.Constant(0)
##action_selection = dqn.backend.EpsilonGreedy(epsilon_scheduler)
##loss = dqn.backend.TDErrorLoss()

# Categorical Algorithm (C51)
##q_func = nn.DistributionalQNetwork([64], env.action_space.n, n=51, noisy_net=True, dueling=[32])
###q_func = nn.DistributionalQNetwork([64], env.action_space.n, n=51)
###epsilon_scheduler = dqn.annealing_schedules.Linear(start=1, end=0.02, num_steps=10000)
##epsilon_scheduler = dqn.annealing_schedules.Constant(0)
##action_selection = dqn.backend.EpsilonGreedy(epsilon_scheduler)
##loss = dqn.backend.CategoricalAlgorithm()

# Quantile Regression DQN (QR-DQN)
#q_func = nn.DistributionalQNetwork([64], env.action_space.n, n=200, noisy_net=True, dueling=[32])
q_func = nn.DistributionalQNetwork([64], env.action_space.n, n=51)
#epsilon_scheduler = dqn.annealing_schedules.Linear(start=1, end=0.02, num_steps=10000)
epsilon_scheduler = dqn.annealing_schedules.Constant(0)
action_selection = dqn.backend.EpsilonGreedy(epsilon_scheduler)
loss = dqn.backend.QuantileRegressionLoss()

agent = DQNAgent(network=q_func,
                 observation_space_shape=env.observation_space.shape,
                 num_actions=env.action_space.n,
                 action_selection=action_selection,
                 loss=loss,
                 update_target_network_frequency=1000)

#agent.train(env, num_timesteps=100000, render=True)
agent.train(env, num_episodes=5000, render=True)
