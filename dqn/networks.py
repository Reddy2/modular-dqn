import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
import functools

def noisy_fully_connected(inputs, num_outputs, activation_fn=tf.nn.relu, factorised=True, sigma_0=0.5, scope='noisy_fully_connected'):
    # factorised=False is same as independent (put this in doc-string)
    # Mention that epsilon is always resampled.  Perhaps have an option to make the noise a tf.Variable/parameter of the function
    
    with tf.variable_scope(scope):
        # TODO: check if we need an if the shape is > 2 for the below line    
        inputs = layers.flatten(inputs)
        batch_size, p = inputs.get_shape().as_list()
        q = num_outputs
        
        if factorised:
            input_noise = tf.random_normal(shape=[p, 1])
            output_noise = tf.random_normal(shape=[1, q])
            
            f = lambda x: tf.sign(x) * tf.sqrt(tf.abs(x))

            weights_mu = tf.get_variable('weights_mu', shape=[p, q], initializer=tf.random_uniform_initializer(minval=-np.sqrt(1/p), maxval=np.sqrt(1/p)))
            weights_sigma = tf.get_variable('weights_sigma', shape=[p, q], initializer=tf.constant_initializer(sigma_0 / np.sqrt(p)))
            weights_noise = f(input_noise) * f(output_noise) # Same as performing the outer product

            biases_mu = tf.get_variable('biases_mu', shape=[q], initializer=tf.random_uniform_initializer(minval=-np.sqrt(1/p), maxval=np.sqrt(1/p)))
            biases_sigma = tf.get_variable('biases_sigma', shape=[q], initializer=tf.constant_initializer(sigma_0 / np.sqrt(p)))
            biases_noise = f(output_noise)
        else:
            # Note shapes are reversed from paper ([p, q] instead of [q, p]) due to using inputs * weights ([N, p] x [p, q] = [N, q]) instead of weights * inputs
            weights_mu = tf.get_variable('weights_mu', shape=[p, q], initializer=tf.random_uniform_initializer(minval=-np.sqrt(3/p), maxval=np.sqrt(3/p)))
            weights_sigma = tf.get_variable('weights_sigma', shape=[p, q], initializer=tf.constant_initializer(0.017))
            weights_noise = tf.random_normal(shape=[p, q])

            biases_mu = tf.get_variable('biases_mu', shape=[q], initializer=tf.random_uniform_initializer(minval=-np.sqrt(3/p), maxval=np.sqrt(3/p)))
            biases_sigma = tf.get_variable('biases_sigma', shape=[q], initializer=tf.constant_initializer(0.017))
            biases_noise = tf.random_normal(shape=[q])

        weights = weights_mu + weights_sigma * weights_noise
        biases = biases_mu + biases_sigma * biases_noise
        affine_transform = tf.matmul(inputs, weights) + biases

        if activation_fn:
            return activation_fn(affine_transform)
        
        return affine_transform
    

class Network:
    def __init__(self, name, forward):
        self.forward = forward
        self._template = tf.make_template(name, self.forward)

    # TODO: Do we need **kwargs here ? Should we allow it in forward ?
    def __call__(self, *inputs, **kwargs):
        return self._template(*inputs, **kwargs)

    # TODO: Should forward be defined here (at least for clarity?)

    @property
    def global_variables(self):
        global_variables = self._template.global_variables
        if not global_variables:
            raise RuntimeWarning("Network " + self._template.name + " has no global variables.  Make sure to use (call) the network at least once to populate these variables.")

        return global_variables

    
class NetworkSpec:
    def build(self, name):
        network = Network(name, self.forward)
##        network.__dict__.update(self.__dict__)
        return network

    def forward(self, *inputs):
        raise NotImplementedError

    @property
    def type(self):
        raise NotImplementedError

# TODO: Perhaps add a custom build function for each NetworkSpec (similar to what we do for losses), as right now we have some issues with duplicate values (action_space in NAF, for example)
# TODO: Add convolutions
# TODO: Dueling only needs to use the advantage stream to evaluate the best q-value for exploration (see dueling dqn paper)
#       We may be able to do this in the call method by having an "exploration" option or something similar.. seems to get rid of modularity though
class QNetwork(NetworkSpec):
    def __init__(self, hiddens, num_actions, activation_fn=tf.nn.relu, noisy_net=False, dueling=None):
        self._hiddens = hiddens
        self.num_actions = num_actions
        self.activation_fn = activation_fn
        self.noisy_net = noisy_net
        self.dueling = dueling

## TODO: Dueling with separate number of hiddens for value/advantage streams
##        if dueling and np.array(dueling).ndim == 1:
##            dueling = np.tile(, (2, 1))

##        self.value_hiddens = []
##        self.advantage_hiddens = []
##        if dueling:
##            if np.array(dueling).ndim == 1:
##                self.value_hiddens = dueling
##                self.advantage_hiddens = dueling
##            else:
##                self.value_hiddens, self.advantage_hiddens = dueling
                
        self._layer = layers.fully_connected
        if noisy_net:
            self._layer = noisy_fully_connected

    def forward(self, state):
        out = state
        for index, hidden in enumerate(self._hiddens):
            out = self._layer(inputs=out, num_outputs=hidden, activation_fn=self.activation_fn, scope='fully_connected' + str(index))

        if self.dueling is not None:   # Note: [] is allowed (so don't use if self.dueling:)
            value = out
            for index, hidden in enumerate(self.dueling):
                value = self._layer(inputs=value, num_outputs=hidden, activation_fn=self.activation_fn, scope='value_fully_connected' + str(index))
            value = self._layer(inputs=out, num_outputs=1, activation_fn=None, scope='value_output_fully_connected')

            advantage = out
            for index, hidden in enumerate(self.dueling):
                advantage = self._layer(inputs=advantage, num_outputs=hidden, activation_fn=self.activation_fn, scope='advantage_fully_connected' + str(index))
            advantage = self._layer(inputs=out, num_outputs=self.num_actions, activation_fn=None, scope='advantage_output_fully_connected')
            advantage_mean = tf.reduce_mean(advantage, axis=1, keepdims=True)

            out = value + advantage - advantage_mean
        else:
            out = self._layer(inputs=out, num_outputs=self.num_actions, activation_fn=None, scope='output_fully_connected')
            
        return out

    @property
    def type(self):
        return 'q_network'


class DistributionalQNetwork(NetworkSpec):
    def __init__(self, hiddens, num_actions, n, activation_fn=tf.nn.relu, noisy_net=False, dueling=None):
        # (for now) n represents either the number of atoms (categorical) or number of quantiles (quantile regression)
        self._hiddens = hiddens
        self.num_actions = num_actions
        self.n = n
        self.activation_fn = activation_fn
        self.noisy_net = noisy_net
        self.dueling = dueling

        self._layer = layers.fully_connected
        if noisy_net:
            self._layer = noisy_fully_connected

    def forward(self, state):
        out = state
        for index, hidden in enumerate(self._hiddens):
            out = self._layer(inputs=out, num_outputs=hidden, activation_fn=self.activation_fn, scope='fully_connected' + str(index))

        if self.dueling is not None:   # Note: [] is allowed (so don't use if self.dueling:)
            value = out
            for index, hidden in enumerate(self.dueling):
                value = self._layer(inputs=value, num_outputs=hidden, activation_fn=self.activation_fn, scope='value_fully_connected' + str(index))
            value = self._layer(inputs=out, num_outputs=self.n, activation_fn=None, scope='value_output_fully_connected')
            value = tf.expand_dims(value, axis=1)
            
            advantage = out
            for index, hidden in enumerate(self.dueling):
                advantage = self._layer(inputs=advantage, num_outputs=hidden, activation_fn=self.activation_fn, scope='advantage_fully_connected' + str(index))
            advantage = self._layer(inputs=out, num_outputs=self.num_actions * self.n , activation_fn=None, scope='advantage_output_fully_connected')
            advantage = tf.reshape(advantage, [-1, self.num_actions, self.n])
            advantage_mean = tf.reduce_mean(advantage, axis=1, keepdims=True)

            logits = value + advantage - advantage_mean
        else:
            out = self._layer(inputs=out, num_outputs=self.num_actions * self.n, activation_fn=None, scope='fully_connected' + str(index + 1))
            logits = tf.reshape(out, [-1, self.num_actions, self.n])

        # TODO: Put below in doc-string
        # Note: We output logits, not probabilities
        #       For the categorical algorithm we don't use tf.nn.softmax for the activation_fn for numerical stability (and possibly computation time) issues.  Instead later on we will use tf.nn.softmax_cross_entropy_with_logits_v2
        return logits

    @property
    def type(self):
        return 'distributional'


class NAFQNetwork(NetworkSpec):
    # TODO: Explain shared_hiddens behavior/diagonal_l
    # TODO: Document that if overwritten for a custom network, self._diagonal_l must be specified
    def __init__(self, mu_hiddens, value_hiddens, l_hiddens, action_space_shape, shared_hiddens=None, activation_fn=tf.nn.relu, diagonal_l=False, noisy_net=False):
        self._mu_hiddens = mu_hiddens
        self._value_hiddens = value_hiddens
        self._l_hiddens = l_hiddens
        self.action_space_shape = action_space_shape
        self._shared_hiddens = shared_hiddens
        self._activation_fn = activation_fn
        
        self._layer = layers.fully_connected
        if noisy_net:
            self._layer = noisy_fully_connected

        self._diagonal_l = diagonal_l
        self.num_l_outputs = action_space_shape[0] if diagonal_l else action_space_shape[0] * (action_space_shape[0] + 1) // 2

    def forward(self, state):
        out = state
        
        if self._shared_hiddens is not None:   # Note: [] is allowed (so don't use if self._shared_hiddens):            
            for index, hidden in enumerate(self._shared_hiddens):
                out = self._layer(inputs=out, num_outputs=hidden, activation_fn=self._activation_fn, scope='fully_connected' + str(index))
            
        for index, hidden in enumerate(self._mu_hiddens):
            mu = self._layer(inputs=out, num_outputs=hidden, activation_fn=self._activation_fn, scope='mu_fully_connected' + str(index))
        mu = self._layer(inputs=out, num_outputs=self.action_space_shape[0], activation_fn=None, scope='mu_output_fully_connected')

        for index, hidden in enumerate(self._value_hiddens):
            value = self._layer(inputs=out, num_outputs=hidden, activation_fn=self._activation_fn, scope='value_fully_connected' + str(index))
        value = self._layer(inputs=out, num_outputs=1, activation_fn=None, scope='value_output_fully_connected')
        value = tf.squeeze(value)

        for index, hidden in enumerate(self._l_hiddens):
            flat_l = self._layer(inputs=out, num_outputs=hidden, activation_fn=self._activation_fn, scope='l_fully_connected' + str(index))
        flat_l = self._layer(inputs=out, num_outputs=self.num_l_outputs, activation_fn=None, scope='l_output_fully_connected')
        
        return mu, value, flat_l

    @property
    def diagonal_l(self):
        return self._diagonal_l
        
    @property
    def type(self):
        return 'naf'
