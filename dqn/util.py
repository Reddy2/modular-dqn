import tensorflow as tf

# A possibly (much) better implementation of this is located at
# https://github.com/keras-team/keras/blob/master/keras/backend/tensorflow_backend.py
# under class Function
# TODO: One thing of note is they use tf.control_dependencies(outputs) on the updates
# This would probably be a good thing for us to add

class TFFunction:
    # TODO: Document the function.  Document that inputs/outputs doesn't need to be a list
    def __init__(self, inputs, outputs, updates=[], givens={}, sess=None):
        self.sess = sess
        self._inputs = inputs

        if not isinstance(self._inputs, (list, tuple)):
            self._inputs = [self._inputs]

        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]

        # TODO: Maybe handle when updates has 0 elements and 1 element
        #     Using tf.group() on 0 elements may be confusing in TensorBoard ?  Have to test
        self._ops_to_run = list(outputs) + [tf.group(*updates)]
        self._givens = givens

    # Cannot do traditional **kwargs since they must be passed as strings and we need objects
    def __call__(self, *args, kwargs={}):
        if self.sess is None:
            self.sess = tf.get_default_session()
        
        feed_dict = {}

        for inpt, value in kwargs.items():
            feed_dict[inpt] = value

        for inpt, value in zip(self._inputs, args):
            feed_dict[inpt] = value

        for inpt in self._givens:
            feed_dict[inpt] = feed_dict.get(inpt, self._givens[inpt])

        outputs = self.sess.run(self._ops_to_run, feed_dict=feed_dict)[:-1]
        
        if len(outputs) == 1:
            return outputs[0]
        
        return outputs
