import tensorflow as tf
import numpy as np

def euclidean_loss(y_true, y_pred):
    return tf.reduce_mean(tf.reduce_sum(tf.pow(y_true - y_pred, 2))) / tf.cast(tf.shape(y_true)[0], 'float')


def MSE(a, b):
    return np.mean((a - b) ** 2)


def eval_class_acc(y_hat_logits, y_true_labels):
    logit_arg_max = np.argmax(y_hat_logits, axis=1)
    y_true_arg_max = np.argmax(y_true_labels, axis=1)
    acc = np.equal(logit_arg_max, y_true_arg_max)
    return np.mean(acc)


def init_tf():
    init = tf.global_variables_initializer()
    init_l = tf.local_variables_initializer()
    sess = tf.get_default_session()
    # Run the initializer
    sess.run(init)
    sess.run(init_l)