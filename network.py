#!/usr/bin/python

import tensorflow as tf
import numpy as np

from flip_gradient import *
import embed_network as en
import embed_network_music as enm
import embed_network_video as env
import embed_loss as el
import eval

import OPTS


class Model:
    class OPTS(OPTS.OPTS):
        def __init__(self):
            OPTS.OPTS.__init__(self, 'Model OPTS')
            self.network_name = None
            self.x_dim = None
            self.y_dim = None
            self.x_num_layer = 2
            self.y_num_layer = 2
            self.constraint_weights = [2, 1]
            self.batch_size = 1024
            self.is_linear = False

    def __init__(self, opts):
        if opts is None:
            opts = self.OPTS()
        self.opts = opts
        self.opts.assert_all_keys_valid()

    def construct(self):

        self.x_data = tf.placeholder(tf.float32, [None, self.opts.x_dim], name='X_data')
        self.y_data = tf.placeholder(tf.float32, [None, self.opts.y_dim], name='Y_data')

         # Build embedding
        self.aff_xy = tf.placeholder(tf.bool, [None, None], name='aff_xy')
        self.K = tf.placeholder(tf.int32, name='K')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.l = tf.placeholder(tf.float32, name='lambda_for_adv')

        x_net_opts = enm.Music_Model.OPTS()
        x_net_opts.network_name = 'X_network'
        x_net_opts.num_layer = self.opts.x_num_layer
        self.x_net = enm.Music_Model(x_net_opts)

        if(self.opts.x_num_layer == 3 and self.opts.y_num_layer == 2):
            y_net_opts = en.Wang_Model.OPTS()
            y_net_opts.network_name = 'Y_network'
            self.y_net = en.Wang_Model(y_net_opts)
        else:
            y_net_opts = env.Video_Model.OPTS()
            y_net_opts.network_name = 'Y_network'
            y_net_opts.num_layer = self.opts.y_num_layer
            self.y_net = env.Video_Model(y_net_opts)

        self.x_embed = self.x_net.construct(self.x_data, keep_prob=self.keep_prob, is_linear=self.opts.is_linear, is_training=self.is_training)
        self.y_embed = self.y_net.construct(self.y_data, keep_prob=self.keep_prob, is_linear=self.opts.is_linear, is_training=self.is_training)



        # Cross-modal Triplet loss
        el_opts = el.Triplet.OPTS()
        el_opts.network_name = 'Triplet'
        el_opts.distance = el_opts.DIST.COS

        el_net = el.Triplet(el_opts)
        self.loss_xy, self.loss_yx = el_net.construct(self.x_embed, self.y_embed, self.aff_xy, self.K, 'Triplet_Net')

        # Adversarial loss
        self.concat_feat = tf.concat(0, [self.x_embed, self.y_embed])
        self.dom_label = tf.concat(0, [tf.tile([0.], [tf.shape(self.x_embed)[0]]),
                                       tf.tile([1.], [tf.shape(self.y_embed)[0]])])
        self.dom_label = tf.expand_dims(self.dom_label, 1)
        self.dom_loss, self.dom_acc = self.dom_classifier(self.concat_feat, self.dom_label, self.l, self.keep_prob)

        self.loss = self.loss_xy * self.opts.constraint_weights[0] + self.loss_yx * self.opts.constraint_weights[1]

        # calculate gradients
        _, self.xy_rank_x = tf.nn.moments(tf.gradients(self.loss_xy, [self.x_embed])[0], [0, 1])
        _, self.yx_rank_x = tf.nn.moments(tf.gradients(self.loss_yx, [self.x_embed])[0], [0, 1])
        _, self.dom_x = tf.nn.moments(tf.gradients(self.dom_loss, [self.x_embed])[0], [0, 1])
        _, self.xy_rank_y = tf.nn.moments(tf.gradients(self.loss_xy, [self.y_embed])[0], [0, 1])
        _, self.yx_rank_y = tf.nn.moments(tf.gradients(self.loss_yx, [self.y_embed])[0], [0, 1])
        _, self.dom_y = tf.nn.moments(tf.gradients(self.dom_loss, [self.y_embed])[0], [0, 1])



        eval_opts = eval.Recall.OPTS()
        eval_opts.network_name = 'Recall'

        eval_net = eval.Recall(eval_opts)

        self.recall_xy, self.recall_yx, self.xy_idx, self.yx_idx = eval_net.construct(self.x_embed, self.y_embed,
                                                                                      self.aff_xy,
                                                                                      [1, 10, 25, 50, 100, 1000])

        # self.debug_list = el_net.debug_list

    def dom_classifier(self, tensor, labels, l=1., keep_prob=1.):
        with tf.name_scope("dom_classifier"):
            feature = flip_gradient(tensor, l)

            d_fc1 = tf.nn.relu(self.fc_layer(feature, 512, "dom_fc1"))
            d_fc1 = tf.nn.dropout(d_fc1, keep_prob)

            d_fc2 = tf.nn.relu(self.fc_layer(d_fc1, 512, "dom_fc2"))
            d_fc2 = tf.nn.dropout(d_fc2, keep_prob)

            # d_logits = self.fc_layer(d_fc2, 2, "dom_logits")
            d_logit = self.fc_layer(d_fc2, 1, "dom_logit")

        with tf.name_scope("domain_acc_and_loss"):
            # self.domain_prediction = tf.nn.softmax(d_logits)
            domain_prediction = tf.sigmoid(d_logit)

            # self.domain_loss = tf.nn.softmax_cross_entropy_with_logits(d_logits, self.domain)
            domain_loss = tf.nn.sigmoid_cross_entropy_with_logits(d_logit, labels)
            domain_loss = tf.reduce_mean(domain_loss)

            # domain acc
            correct_domain_prediction = tf.equal(tf.round(domain_prediction), labels)
            domain_acc = tf.reduce_mean(tf.cast(correct_domain_prediction, tf.float32))
        return domain_loss, domain_acc

    def fc_layer(self, tensor, dim, name, reuse=False):
        in_shape = tensor.get_shape()

        with tf.variable_scope(name, reuse=reuse):
            weights = tf.get_variable("weights", shape=[in_shape[1], dim],
                                      initializer=tf.contrib.layers.xavier_initializer())
            biases = tf.get_variable("biases", shape=[dim],
                                     initializer=tf.constant_initializer(0.0))
            fc = tf.nn.xw_plus_b(tensor, weights, biases)
            return fc

