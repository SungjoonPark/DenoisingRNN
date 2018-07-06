import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import RNNCell
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import init_ops

class DGRUD_cell(RNNCell):

    def __init__(self, n_of_features, t, hidden_size):
        self.hidden_size = hidden_size
        self.n_of_features = n_of_features
        self.time = t

    def __call__(self, inputs, state, scope=None):
        h, d, last_observation, prev_imputed_input, prev_stored_values = state

        with vs.variable_scope(scope or "denoising_gru_d_cell"):

            with vs.variable_scope("input_processing"):
                # pre_processing
                m = tf.cast( tf.equal( tf.is_nan(inputs), False ), tf.float32) # build mask
                inputs = tf.where(tf.is_nan(inputs), tf.zeros_like(inputs), inputs) # np.nan_to_num(inp)

                # kernels
                mean_kernel = vs.get_variable("mean_kernel", shape=[1, self.time, 1], dtype=tf.float32, initializer=tf.truncated_normal_initializer(0., .01))

                # smoothing last obs
                prev_stored_values = tf.reshape(prev_stored_values, shape=[-1, self.time, self.n_of_features])
                smoothed_last_observation = tf.reduce_mean(mean_kernel * prev_stored_values, 1)

                # smoothing cur obs
                new_last_observation = (1. - m) * last_observation + m * inputs
                new_stored_values = tf.concat((prev_stored_values[:,1:,:], tf.reshape(new_last_observation, shape=[-1, 1, self.n_of_features])), 1)
                smoothed_cur_observation = tf.reduce_mean(mean_kernel * new_stored_values, 1)

                # decay rate for x
                W_x = vs.get_variable("decay_W", shape=[self.n_of_features, self.n_of_features], dtype=tf.float32)
                b_x = vs.get_variable("decay_b", shape=[self.n_of_features], dtype=tf.float32)
                gamma_x = tf.exp(-tf.nn.relu(tf.matmul(d, W_x) + b_x))

                # input for current timestep
                X = m * smoothed_cur_observation + (1. - m) * gamma_x * smoothed_cur_observation #zero state as a default

                # for next iter
                new_stored_values = tf.reshape(new_stored_values, shape=[-1, self.time*self.n_of_features])

            with vs.variable_scope("gates"):
                ihm = tf.concat((X, h, m), 1)

                W_g = vs.get_variable("kernel", shape=[ihm.get_shape()[1].value, 2*self.hidden_size], dtype=tf.float32)
                b_g = vs.get_variable("bias", shape=[2*self.hidden_size], dtype=tf.float32, initializer=init_ops.constant_initializer(1.0, dtype=tf.float32))

                concat = tf.sigmoid(tf.matmul(ihm, W_g) + b_g)
                r, z = tf.split(concat, num_or_size_splits=2, axis=1)

            with vs.variable_scope("candidate"):
                #candidate_h
                irhm = tf.concat((X, r * h, m), 1)

                W_h = vs.get_variable("kernel", shape=[irhm.get_shape()[1].value, self.hidden_size], dtype=tf.float32)
                b_h = vs.get_variable("bias", shape=[self.hidden_size], dtype=tf.float32)

                cand_h = tf.tanh(tf.matmul(irhm, W_h) + b_h)

                # decay rate for h
                W_h = vs.get_variable("decay_W", shape=[self.n_of_features, self.hidden_size], dtype=tf.float32)
                b_h = vs.get_variable("decay_b", shape=[self.hidden_size], dtype=tf.float32)
                gamma_h = tf.exp(-tf.nn.relu(tf.matmul(d, W_h) + b_h))

                # new_h (adding decay term gamma_h to prev_h)
                new_h = z * (gamma_h * h) + (1.0 - z) * cand_h

            # update delta: add one, then add previous d to only missing values
            new_d = tf.ones_like(d) + (1-m) * d
            next_input = tf.concat([X, new_h], axis=1)

        return next_input, (new_h, new_d, new_last_observation, X, new_stored_values)

    @property
    def state_size(self):
        return self.hidden_size, self.n_of_features, self.n_of_features, self.n_of_features, self.time*self.n_of_features

    @property
    def output_size(self):
        return self.n_of_features + self.hidden_size
