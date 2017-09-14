#!/usr/bin/python

import tensorflow as tf
import numpy as np

import OPTS


class Music_Model:
    class OPTS(OPTS.OPTS):
        def __init__(self):
            OPTS.OPTS.__init__(self, 'Music_Model OPTS')
            self.network_name = None
            self.num_layer = 2

    def __init__(self, opts=None):
        if opts is None:
            opts = self.OPTS()
        self.opts = opts
        self.opts.assert_all_keys_valid()
        self.__is_constructed = False
        self.is_training = True  # move into opts!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


    def construct(self, x_data, keep_prob, is_linear = False, is_training=None):
        with tf.variable_scope(self.opts.network_name):
            self.is_training = is_training
            self.fc1 = self.fc_layer(x_data, 2048, name='fc1')
            self.relu1 = tf.nn.relu(self.fc1, name='relu1')
            if (is_linear == False):
                self.relu1_drop = tf.nn.dropout(self.relu1, keep_prob=keep_prob, name='relu1_drop')
            else:
                self.relu1_drop = tf.nn.dropout(self.fc1, keep_prob=keep_prob, name='fc1_drop')

            self.fc2 = self.fc_layer(self.relu1_drop, 1024, name='fc2')
            self.relu2 = tf.nn.relu(self.fc2, name='relu2')
            if (is_linear == False):
                self.relu2_drop = tf.nn.dropout(self.relu2, keep_prob=keep_prob, name='relu2_drop')
            else:
                self.relu2_drop = tf.nn.dropout(self.fc2, keep_prob=keep_prob, name='fc2_drop')

            if (self.opts.num_layer == 3): # We only consier that the number of layers for music is only 3...
                self.relu_last = self.relu2_drop
            else:
                self.fc3 = self.fc_layer(self.relu2_drop, 1024, name='fc3')
                self.relu3 = tf.nn.relu(self.fc3, name='relu3')
                if (is_linear == False):
                    self.relu3_drop = tf.nn.dropout(self.relu3, keep_prob=keep_prob, name='relu3_drop')
                else:
                    self.relu3_drop = tf.nn.dropout(self.fc3, keep_prob=keep_prob, name='fc3_drop')

                self.fc4 = self.fc_layer(self.relu3_drop, 512, name='fc4')
                self.relu4 = tf.nn.relu(self.fc4, name='relu4')
                if (is_linear == False):
                    self.relu4_drop = tf.nn.dropout(self.relu4, keep_prob=keep_prob, name='relu4_drop') #orignally I mistake relu4_drop -> relu_3_drop...
                else:
                    self.relu4_drop = tf.nn.dropout(self.fc4, keep_prob=keep_prob, name='fc4_drop')

                if (self.opts.num_layer == 1):
                    self.relu_last = x_data
                elif (self.opts.num_layer == 2):
                    self.relu_last = self.relu1_drop
                elif (self.opts.num_layer == 4):
                    self.relu_last = self.relu3_drop
                elif (self.opts.num_layer == 5):
                    self.relu_last = self.relu4_drop
                else:
                    print ("invalid layer, num layer should be 2~5")
                    return

            self.fc5 = self.fc_layer(self.relu_last, 512, name='fc5')
            # self.bn = self.batch_normalization(self.fc2, 512, name='batch_normalize')
            self.bn = tf.contrib.layers.batch_norm(inputs=self.fc5, decay=0.99, center=True, scale=True, epsilon=1e-3,
                                                   is_training=self.is_training, scope='BN')
            self.l2_norm = tf.nn.l2_normalize(self.bn, dim=1, name='l2_normalize')
            return self.l2_norm

    def fc_layer(self, bottom, output_dim, name, reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = tf.get_variable(name='W', shape=[dim, output_dim],
                                      initializer=self.initializeWeight(shape=[dim, output_dim]))
            # fc = tf.matmul(x, weights)  # to avoid conflict from beta of  batch normalization
            bias = tf.get_variable(name='b', shape=[output_dim],
                                   initializer=tf.constant_initializer(value=0., dtype=tf.float32))
            fc = tf.nn.xw_plus_b(x, weights, bias, name="fc")
            return fc

    # http://r2rt.com/implementing-batch-normalization-in-tensorflow.html
    def batch_normalization(self, x, dim, name, reuse=False):
        with tf.variable_scope(name):
            batch_mean, batch_var = tf.nn.moments(x, [0])

            scale = tf.get_variable(name='scale', shape=[dim],
                                    initializer=tf.constant_initializer(value=1, dtype=tf.float32))
            beta = tf.get_variable(name='beta', shape=[dim],
                                   initializer=tf.constant_initializer(value=0, dtype=tf.float32))

            BN = tf.nn.batch_normalization(x, batch_mean, batch_var, beta, scale, 1e-3)
            return BN

    # xavier
    def initializeWeight(self, shape, method='xavier2'):
        inputsize = reduce(lambda x, y: x * y, shape[0:-1])
        if method == 'xavier2':
            stddev = np.sqrt(2.0 / (inputsize))
        else:
            outsize = shape[-1]
            stddev = np.sqrt(3.0 / (inputsize + outsize))
        return tf.truncated_normal_initializer(stddev=stddev)
