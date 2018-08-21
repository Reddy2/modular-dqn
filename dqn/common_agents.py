import dqn.agent

class CategoricalAlgorithm(dqn.agent.DQNAgent):
    def __init__(self,
                 network,      # Requires a DistributionalQNetwork
                 observation_space_shape,
                 num_actions,
                 memory,
                 epsilon_scheduler,
                 batch_size=32,
                 n_step=3,
                 discount_factor=1.0,
                 replay_period=1,
                 update_target_network_frequency=500):
        
        action_selection = dqn.backend.EpsilonGreedy(epsilon_scheduler)
        loss = dqn.backend.CategoricalAlgorithm()

        super().__init__(network=network,
                         observation_space_shape=observation_space_shape,
                         num_actions=num_actions,
                         action_selection=action_selection,
                         loss=loss,
                         memory=memory,
                         batch_size=batch_size,
                         n_step=n_step,
                         discount_factor=discount_factor,
                         replay_period=replay_period,
                         update_target_network_frequency=update_target_network_frequency)

class QuantileRegression(dqn.agent.DQNAgent):
    def __init__(self,
                 network,      # Requires a DistributionalQNetwork
                 observation_space_shape,
                 num_actions,
                 memory,
                 epsilon_scheduler,
                 batch_size=32,
                 n_step=3,
                 discount_factor=1.0,
                 replay_period=1,
                 update_target_network_frequency=500):
        
        action_selection = dqn.backend.EpsilonGreedy(epsilon_scheduler)
        loss = dqn.backend.QuantileRegressionLoss()

        super().__init__(network=network,
                         observation_space_shape=observation_space_shape,
                         num_actions=num_actions,
                         action_selection=action_selection,
                         loss=loss,
                         memory=memory,
                         batch_size=batch_size,
                         n_step=n_step,
                         discount_factor=discount_factor,
                         replay_period=replay_period,
                         update_target_network_frequency=update_target_network_frequency)
