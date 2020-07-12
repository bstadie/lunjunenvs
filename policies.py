
import numpy as np
import tensorflow as tf
from utils import euclidean_loss, MSE, eval_class_acc
import tensorflow.contrib.layers as L
from baselines.common.tf_util import function


class RandomPol:
    def __init__(self, env):
        self.env = env

    def train(self, replay_buffer, num_timesteps):
        pass

    def act(self, state):
        a = np.random.randn(self.env.action_dim)
        return a, a



class BCPol:
    def __init__(self, s_mb, a_mb, discrete=False):
        self.s_shp = s_mb.shape
        self.a_shp = a_mb.shape
        self.discrete = discrete
        self.nn_input_s = tf.placeholder(dtype=tf.float32, shape=[None, self.s_shp[1]])
        self.nn_input_a = tf.placeholder(dtype=tf.float32, shape=[None, self.a_shp[1]])
        self.make_model()

    def make_model(self):
        h = L.fully_connected(self.nn_input_s, 32)
        h = L.fully_connected(h, 32, activation_fn=tf.nn.tanh)
        self.out = L.fully_connected(h, self.a_shp[1], activation_fn=tf.identity)
        if self.discrete is True:
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.nn_input_a, logits=self.out)
            self.out = tf.nn.softmax(self.out)
        if self.discrete is False:
            loss = euclidean_loss(y_true=self.nn_input_a, y_pred=self.out)

        update = tf.train.AdamOptimizer().minimize(loss)

        self.output_fn = function(inputs=[self.nn_input_s], outputs=[self.out])
        self.train_fn = function(inputs=[self.nn_input_s, self.nn_input_a], outputs=[loss], updates=[update])

    def train(self, s_mb, a_mb):
        loss = self.train_fn(s_mb, a_mb)
        return np.mean(loss[0])

    def eval(self, s_mb, a_mb):
        if self.discrete is False:
            a_hat = self.act(s_mb)
            return MSE(a_hat, a_mb)
        else:
            a_hat = self.act(s_mb)
            return eval_class_acc(a_hat, a_mb)

    def act(self, state):
        #print(state.shape)
        a = self.output_fn(state)
        a = a[0]
        if self.discrete is True:
            #a = np.argmax(a, axis=1)
            return a
        else:
            return a