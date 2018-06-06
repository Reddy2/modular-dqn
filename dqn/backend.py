import tensorflow as tf
import numpy as np
import functools
from dqn.util import TFFunction

### Action Selection ###

def epsilon_greedy_td_error(online_net, observation_space_shape, num_actions):
    with tf.variable_scope('act'):
        state = tf.placeholder(tf.float32, shape=list(observation_space_shape), name='state')
        epsilon = tf.placeholder(tf.float32, shape=[])   # Must be 0 <= eps <= 1.. perhaps enforce this somehow ?!

        expanded_state = tf.expand_dims(state, axis=0)        
        expected_q_values = online_net(expanded_state)
        greedy_action = tf.argmax(expected_q_values, axis=1)[0]
        
        random_action = tf.random_uniform(shape=[], minval=0, maxval=num_actions, dtype=tf.int64)
        action = tf.where(tf.random_uniform(shape=[], minval=0, maxval=1, dtype=tf.float32) < epsilon,
                          random_action, greedy_action)

        return TFFunction(inputs=[state, epsilon],
                          outputs=action)


def epsilon_greedy_categorical_algorithm(online_net, observation_space_shape, num_actions, num_atoms, v_min, v_max):
    with tf.variable_scope('act'):
        state = tf.placeholder(tf.float32, shape=list(observation_space_shape), name='state')
        epsilon = tf.placeholder(tf.float32, shape=[])   # Must be 0 <= eps <= 1.. perhaps enforce this somehow ?!

        expanded_state = tf.expand_dims(state, axis=0)
        atoms = tf.linspace(float(v_min), float(v_max), num_atoms)
        logits = online_net(expanded_state)
        prob_distributions = tf.nn.softmax(logits)
        expected_q_values = tf.reduce_sum(atoms * prob_distributions, axis=2)
        greedy_action = tf.argmax(expected_q_values, axis=1)[0]
        
        random_action = tf.random_uniform(shape=[], minval=0, maxval=num_actions, dtype=tf.int64)
        action = tf.where(tf.random_uniform(shape=[], minval=0, maxval=1, dtype=tf.float32) < epsilon,
                          random_action, greedy_action)

        return TFFunction(inputs=[state, epsilon],
                          outputs=action)


def epsilon_greedy_quantile_regression(online_net, observation_space_shape, num_actions, num_quantiles):
    with tf.variable_scope('act'):
        state = tf.placeholder(tf.float32, shape=list(observation_space_shape), name='state')
        epsilon = tf.placeholder(tf.float32, shape=[])   # Must be 0 <= eps <= 1.. perhaps enforce this somehow ?!

        expanded_state = tf.expand_dims(state, axis=0)
        quantiles = online_net(expanded_state)
        # TODO: We can probably remove 1/num_quantiles since we should get the same argmax
        expected_q_values = 1/num_quantiles * tf.reduce_sum(quantiles, axis=2)
        greedy_action = tf.argmax(expected_q_values, axis=1)[0]
        
        random_action = tf.random_uniform(shape=[], minval=0, maxval=num_actions, dtype=tf.int64)
        action = tf.where(tf.random_uniform(shape=[], minval=0, maxval=1, dtype=tf.float32) < epsilon,
                          random_action, greedy_action)

        return TFFunction(inputs=[state, epsilon],
                          outputs=action)
    

class EpsilonGreedy:
    def __init__(self, epsilon_annealing_schedule):
        self.annealing_schedule = epsilon_annealing_schedule

    def build(self, agent):
        # TODO: Add error handling if loss is not defined
        loss_type = agent.loss_spec.type

        if loss_type == 'td_error':
            func = epsilon_greedy_td_error(agent.online_net, agent.observation_space_shape, agent.num_actions)
        elif loss_type == 'categorical_algorithm':
            func = epsilon_greedy_categorical_algorithm(agent.online_net, agent.observation_space_shape, agent.num_actions, agent.network_spec.n, agent.loss_spec.v_min, agent.loss_spec.v_max) 
        elif loss_type == 'quantile_regression':
            func = epsilon_greedy_quantile_regression(agent.online_net, agent.observation_space_shape, agent.num_actions, agent.network_spec.n)
        else:
            raise NotImplementedError("Loss type '" + loss_type + "' is not supported for epsilon greedy")

        def act(state):
            epsilon = self.annealing_schedule.value(agent.num_acts)
            return func(state, epsilon)
        return act
            

### Update Target ###

def build_update_target(q_online_vars, q_target_vars):
    ## TODO: Perhaps add in a variable scope
    updates = []
    for q_online_var, q_target_var in zip(sorted(q_online_vars, key=lambda var: var.name),
                                          sorted(q_target_vars, key=lambda var: var.name)):
        updates.append(q_target_var.assign(q_online_var))

    return TFFunction(inputs=[], outputs=[], updates=updates)


class UpdateTarget:
    def build(self, agent):
        return build_update_target(agent.online_net.global_variables, agent.target_net.global_variables)


### DQN priorities and loss optimizers ###


def tf_huber_loss(x, delta=1.0):
    return tf.where(tf.abs(x) <= delta,
                    0.5 * tf.square(x),
                    delta * (tf.abs(x) - 0.5*delta))

# For traditional Q Network (non-distributional)
# TODO: Add default parameters
def build_td_error_loss(observation_space_shape, num_actions, q_online, q_target, optimizer=None, double_q=True):
    # Note: gammas_n expected to be 0 for terminal transitions
    with tf.variable_scope('td_error'):
        states_t = tf.placeholder(tf.float32, shape=[None] + list(observation_space_shape), name='state_t')
        actions_t = tf.placeholder(tf.int32, shape=[None], name='action_t')
        rewards_tn = tf.placeholder(tf.float32, shape=[None], name='reward_tn')
        states_tpn = tf.placeholder(tf.float32, shape=[None] + list(observation_space_shape), name='state_tpn')
        gammas_n = tf.placeholder(tf.float32, shape=[None], name='gammas_n')
        importance_weights = tf.placeholder(tf.float32, shape=[None], name='importance_weights')
        
        q_next_target = q_target(states_tpn)
        if double_q:
            # Q_target(S_(t+1), argmax_a[Q_online(S_(t+1), a)])
            q_next_online = q_online(states_tpn)
            greedy_actions_online = tf.argmax(q_next_online, axis=1)
            q_next = tf.reduce_sum(q_next_target * tf.one_hot(greedy_actions_online, num_actions), axis=1)
        else:
            # max_a[Q_target(S_(t+1), a)]
            q_next = tf.reduce_max(q_next_target, axis=1)

        targets = rewards_tn + gammas_n * q_next
        predictions = tf.reduce_sum(q_online(states_t) * tf.one_hot(actions_t, num_actions), axis=1)
        td_errors = tf.stop_gradient(targets) - predictions

        # Perhaps make huber loss an option (and especially the delta an option !)
        #TODO: Explain the use of Huber Loss, referencing the original paper
        #losses = tf.losses.huber_loss(labels=tf.stop_gradient(targets), predictions=predictions, reduction=tf.losses.Reduction.NONE)
        losses = tf_huber_loss(td_errors)
        weighted_loss = tf.reduce_mean(importance_weights * losses)

        # TODO: Gradient clipping (recommended by dueling paper)
        if optimizer is None:
            optimizer = tf.train.AdamOptimizer()
        optimize_op = optimizer.minimize(weighted_loss, var_list=q_online.global_variables)

        return TFFunction(inputs=[states_t, actions_t, rewards_tn, states_tpn, gammas_n, importance_weights],
                          outputs=td_errors,
                          updates=[optimize_op])

# For distributional nets
def build_categorical_algorithm(z_online, z_target, observation_space_shape, v_min=-10, v_max=10, num_atoms=51, optimizer=None, double_q=True):
    # Note: Expects gamma_t = 0 for terminal transitions
    # Note: tpn stands for t + n
    with tf.variable_scope('categorical_algorithm'):
        states_t = tf.placeholder(tf.float32, shape=[None] + list(observation_space_shape), name='state_t')
        actions_t = tf.placeholder(tf.int32, shape=[None], name='action')
        rewards_t = tf.placeholder(tf.float32, shape=[None], name='reward')
        states_tpn = tf.placeholder(tf.float32, shape=[None] + list(observation_space_shape), name='state_tpn')
        gammas_t = tf.placeholder(tf.float32, shape=[None], name='gammas')
        importance_weights = tf.placeholder(tf.float32, shape=[None], name='importance_weight')

        atoms = tf.linspace(float(v_min), float(v_max), num_atoms)
        delta_z = (v_max - v_min) / (num_atoms - 1)
        
        batch_size = tf.shape(states_t)[0]
        batch_indexes = tf.range(0, batch_size)

        states_t_logits = z_online(states_t)
        states_t_indexes = tf.stack([batch_indexes, actions_t], axis=-1)  # effectively zips the two tensors ("lists")
        states_t_actions_t_logits = tf.gather_nd(states_t_logits, states_t_indexes)

        target_states_tpn_logits = z_target(states_tpn)
        target_states_tpn_prob_distributions = tf.nn.softmax(target_states_tpn_logits)
        if double_q:
            online_states_tpn_logits = z_online(states_tpn)
            online_states_tpn_prob_distributions = tf.nn.softmax(online_states_tpn_logits)
            online_states_tpn_expected_q_values = tf.reduce_sum(atoms * online_states_tpn_prob_distributions, axis=2)
            states_tpn_greedy_actions = tf.argmax(online_states_tpn_expected_q_values, axis=1, output_type=tf.int32) # later tf.stack requires same dtype on both tensors
        else:
            target_states_tpn_expected_q_values = tf.reduce_sum(atoms * target_states_tpn_prob_distributions, axis=2)
            states_tpn_greedy_actions = tf.argmax(target_states_tpn_expected_q_values, axis=1, output_type=tf.int32) # later tf.stack requires same dtype on both tensors
        states_tpn_indexes = tf.stack([batch_indexes, states_tpn_greedy_actions], axis=-1)
        states_tpn_greedy_actions_prob_distributions = tf.gather_nd(target_states_tpn_prob_distributions, states_tpn_indexes)
        
        rewards_t_reshaped = tf.reshape(rewards_t, [-1, 1])
        gammas_t_reshaped = tf.reshape(gammas_t, [-1, 1])
        bellman_update = tf.clip_by_value(rewards_t_reshaped + gammas_t_reshaped * atoms, v_min, v_max)
        b = (bellman_update - v_min) / delta_z
        l = tf.floor(b)
        u = l + 1        # Don't use tf.ceil because on an integer b, we have floor(b) == ceil(b) and we lose probability mass in the below calculations (they would both contribute 0 probability)
        m_l = states_tpn_greedy_actions_prob_distributions * (u - b)
        m_u = states_tpn_greedy_actions_prob_distributions * (b - l)

        # If b = v_max (a common occurrence due to clipping on v_max), all probability will go to the atom m_(v_max)
        # from the m_l calculation, and 0 probability will go to the element in the m_u calculation (note v_max is an integer, so b = v_max = l).
        # To do this u uses an out-of-bounds index (index = num_atoms), so we need to clip u's index to be in bounds (to [0, num_atoms-1]) for later.
        # Since 0 probability will go to the corresponding m_u atom this is allowed.  Note we have only provided a minimum clip (of 0, which is the lowest atom) because it is a required argument of tf.clip_by_value
        #    Note: Most commonly this b = l = u issue seems to be implemented by doing
        #    u = tf.ceil(l); m_l = states_tpn_greedy_actions_prob_distributions * (u + tf.cast(tf.equal(l, u), tf.float32) - b)
        #    I assume the clipping method used here is a bit faster, but I could be wrong
        u = tf.clip_by_value(u, 0, num_atoms-1) 

        # Convert indicies of each row of size num_atoms to indicies of batch_size * num_atoms
        # TODO: Explain this wizardry and perhaps rename variables for clarity
        row_offsets = num_atoms * batch_indexes
        l, u = tf.cast(l, tf.int32), tf.cast(u, tf.int32)
        indexes = tf.stack([l, u]) + tf.reshape(row_offsets, [-1, 1])
        indexes = tf.reshape(indexes, [-1, 1])
        updates = tf.concat([m_l, m_u], axis=0)
        updates = tf.reshape(updates, [-1])

        m = tf.scatter_nd(indices=indexes, updates=updates, shape=[batch_size * num_atoms])
        m = tf.reshape(m, [batch_size, num_atoms])

        # TODO: Huber loss ?  Gradient clipping (for dueling nets) ?
        kl_losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.stop_gradient(m), logits=states_t_actions_t_logits)
        weighted_loss = tf.reduce_mean(importance_weights * kl_losses)

        # TODO: When an optimizer is passed in, it will have a different variable scope.  How to deal with this ? (Does it matter ? Appears to look fine in tensorboard)
        if optimizer is None:
            optimizer = tf.train.AdamOptimizer()
        optimize_op = optimizer.minimize(weighted_loss, var_list=z_online.global_variables)

        # TODO: We should probably allow importance_weights to have a given value of all 1s for non-prioritized replay
        return TFFunction(inputs=[states_t, actions_t, rewards_t, states_tpn, gammas_t, importance_weights],
                          outputs=kl_losses,
                          updates=[optimize_op])

# For Distributional Nets
# TODO: NOT TESTED AT ALL
def build_quantile_regression_loss(z_online, z_target, observation_space_shape, num_quantiles=200, kappa=1.0, optimizer=None, double_q=True):
    with tf.variable_scope('quantile_regression'):
        states_t = tf.placeholder(tf.float32, shape=[None] + list(observation_space_shape), name='states_t')
        actions_t = tf.placeholder(tf.int32, shape=[None], name='actions_t')
        rewards_t = tf.placeholder(tf.float32, shape=[None], name='rewards_t')
        states_tpn = tf.placeholder(tf.float32, shape=[None] + list(observation_space_shape), name='states_tpn')
        gammas_t = tf.placeholder(tf.float32, shape=[None], name='gammas_t')
        importance_weights = tf.placeholder(tf.float32, shape=[None], name='importance_weight')

        batch_size = tf.shape(states_t)[0]
        batch_indexes = tf.range(0, batch_size)

        states_t_quantiles = z_online(states_t)
        states_t_indexes = tf.stack([batch_indexes, actions_t], axis=-1)  # effectively zips the two tensors ("lists")
        states_t_actions_t_quantiles = tf.gather_nd(states_t_quantiles, states_t_indexes)

        # TODO: We might be able to eliminate 1/num_quantiles when calculating expected_q_values since it shouldn't affect the argmax (and expected_q_values is not used anywhere else)
        target_states_tpn_quantiles = z_target(states_tpn)
        if double_q:
            online_states_tpn_quantiles = z_online(states_tpn)
            online_states_tpn_expected_q_values = 1/num_quantiles * tf.reduce_sum(online_states_tpn_quantiles, axis=2)
            states_tpn_greedy_actions = tf.argmax(online_states_tpn_expected_q_values, axis=1, output_type=tf.int32) # later tf.stack requires same dtype on both tensors
        else:
            target_states_tpn_expected_q_values = 1/num_quantiles * tf.reduce_sum(target_states_tpn_quantiles, axis=2)
            states_tpn_greedy_actions = tf.argmax(target_states_tpn_expected_q_values, axis=1, output_type=tf.int32) # later tf.stack requires same dtype on both tensors
        states_tpn_indexes = tf.stack([batch_indexes, states_tpn_greedy_actions], axis=-1)
        states_tpn_greedy_actions_quantiles = tf.gather_nd(target_states_tpn_quantiles, states_tpn_indexes)        

        rewards_t_reshaped = tf.reshape(rewards_t, [-1, 1])
        gammas_t_reshaped = tf.reshape(gammas_t, [-1, 1])
        bellman_updates = rewards_t_reshaped + gammas_t_reshaped * states_tpn_greedy_actions_quantiles

        # errors[batch][i][j] = theta'(batch, j) - theta(batch, i)
        # errors[batch][i] = [theta'(batch, 0) - theta(batch, i), theta'(batch, 1) - theta(batch, i), ...]
        errors = tf.stop_gradient(bellman_updates[:, tf.newaxis, :]) - states_t_actions_t_quantiles[:, :, tf.newaxis]
        if kappa == 0:
            # TODO: Explain why we use |errors| * |penalties| rather than errors * penalties (in the paper) for kappa = 0
            losses = tf.abs(errors)
        else:
            losses = tf_huber_loss(errors, delta=kappa)  # tf.losses.huber_loss does not seem to support broadcasting the subtraction when calculating error (at time of writing)

        tau_hats = (2 * tf.range(num_quantiles, dtype=tf.float32) + 1) / (2 * num_quantiles)
        penalties = tf.abs(tau_hats[tf.newaxis, :, tf.newaxis] - tf.cast(errors < 0, tf.float32))
        rhos = penalties * losses

        quantile_regression_losses = 1/num_quantiles * tf.reduce_sum(rhos, axis=[1, 2])   # Same as tf.reduce_sum(tf.reduce_mean(rhos, axis=2), axis=1)
        weighted_loss = tf.reduce_mean(importance_weights * quantile_regression_losses)

        if optimizer is None:
            optimizer = tf.train.AdamOptimizer()
        optimize_op = optimizer.minimize(weighted_loss, var_list=z_online.global_variables)

        return TFFunction(inputs=[states_t, actions_t, rewards_t, states_tpn, gammas_t, importance_weights],
                          outputs=quantile_regression_losses,
                          updates=[optimize_op])


# TODO: Probably turn below into abstract class ?  type doesn't need to be overridden with @property, just def type()

class Loss:
    @property
    def type(self):
        raise NotImplementedError


class TDErrorLoss(Loss):
    def __init__(self, double_q=True, optimizer=None):
        self.optimizer = optimizer
        self.double_q = double_q

    def build(self, agent):
        if agent.network_spec.type != 'q_network':
            raise NotImplementedError("TDErrorLoss not implemented (or not compatible) with network type '" + agent.network_spec.type + "'")

        return build_td_error_loss(observation_space_shape=agent.observation_space_shape, num_actions=agent.num_actions, q_online=agent.online_net, q_target=agent.target_net, double_q=self.double_q, optimizer=self.optimizer)

    @property
    def type(self):
        return 'td_error'
        

# TODO: For n argument of DistributionalQNetwork, perhaps allow it to default to None.  Then set it here based on the loss algorithm (51 for CA, 200 for QL)
#       We may want to set num_atoms or num_quantiles here in the loss, and build the net based on the loss parameters
#           This may be dangerous if later we build a net based on loss.  Although we can probably pass either the loss/net to the net/loss
#           It may be better to do this explicitly anyway
# TODO: Perhaps add/use network_spec (like loss_spec) rather than Network (also rename NetworkBuilder to NetworkSpec)
class CategoricalAlgorithm(Loss):
    def __init__(self, v_min=-10, v_max=10, double_q=True, optimizer=None):
        self.v_min = v_min
        self.v_max = v_max
        self.double_q = double_q
        self.optimizer = optimizer
            
    def build(self, agent):
        if agent.network_spec.type != 'distributional':
            raise NotImplementedError("CategoricalAlgorithm not implemented (or not compatible) with network type '" + agent.network_spec.type + "'")

        return build_categorical_algorithm(z_online=agent.online_net, z_target=agent.target_net, observation_space_shape=agent.observation_space_shape, num_atoms=agent.network_spec.n, v_min=self.v_min, v_max=self.v_max, double_q=self.double_q, optimizer=self.optimizer)               

    @property
    def type(self):
        return 'categorical_algorithm'
    

class QuantileRegressionLoss(Loss):
    def __init__(self, optimizer=None, double_q=True, kappa=1.0):
        self.kappa = kappa
        self.double_q = double_q
        self.optimizer = optimizer

    def build(self, agent):
        if agent.network_spec.type != 'distributional':
            raise NotImplementedError("QuantileRegressionLoss not implemented (or not compatible) with network type '" + agent.network_spec.type + "'")

        return build_quantile_regression_loss(z_online=agent.online_net, z_target=agent.target_net, observation_space_shape=agent.observation_space_shape, num_quantiles=agent.network_spec.n, kappa=self.kappa, double_q=self.double_q, optimizer=self.optimizer)

    @property
    def type(self):
        return 'quantile_regression'
