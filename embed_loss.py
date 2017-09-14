#!/usr/bin/python

import tensorflow as tf
import numpy as np

import OPTS

class Triplet:
  class OPTS(OPTS.OPTS):
    class DIST:
      L2 = 0
      COS = 1
      
    def __init__(self):
      OPTS.OPTS.__init__(self,'Triplet OPTS')
      self.network_name = None
      self.margin = 0.1
      self.distance = self.DIST.L2
      self.is_l2_norm_already = True
      self.num_max_pos_pair = 1

  def __init__(self, opts=None):
    if opts is None:
      opts = self.OPTS()
    self.opts = opts
    self.opts.assert_all_keys_valid()
  
  def construct(self, embed_x, embed_y, aff_xy, K, name):
    with tf.variable_scope(name):
      if not self.opts.is_l2_norm_already:
        embed_x = tf.nn.l2_normalize(embed_x, 1)
        embed_y = tf.nn.l2_normalize(embed_y, 1)

      # convert simliarity to distance by (-1) >> high value indicates the distance between the samples is long
      dist_xy = (-1) *tf.matmul(embed_x, embed_y,transpose_a = False, transpose_b = True)
      self.dist_xy = dist_xy

      dist_pos_pair = tf.select(aff_xy, dist_xy, tf.ones_like(dist_xy,dtype=tf.float32)*(-1e+6))
      dist_neg_pair = tf.select(tf.logical_not(aff_xy), dist_xy, tf.ones_like(dist_xy,dtype=tf.float32)*(1e+6))

      # for top violating postive samples
      top_k_pos_pair_xy, _ = tf.nn.top_k(dist_pos_pair, k=self.opts.num_max_pos_pair)
      top_k_pos_pair_yx, _ = tf.nn.top_k(tf.transpose(dist_pos_pair),k=self.opts.num_max_pos_pair)
      top_k_pos_pair_yx = tf.transpose(top_k_pos_pair_yx)

      # for top violating negative samples
      top_k_neg_pair_xy, _ = tf.nn.top_k(tf.negative(dist_neg_pair),k=K)
      top_k_neg_pair_xy = tf.negative(top_k_neg_pair_xy)
      top_k_neg_pair_yx, _ = tf.nn.top_k(tf.transpose(tf.negative(dist_neg_pair)),k=K)
      top_k_neg_pair_yx = tf.negative(tf.transpose(top_k_neg_pair_yx))

      top_k_pos_pair_xy = tf.tile(top_k_pos_pair_xy, [1,K])
      top_k_pos_pair_yx = tf.tile(top_k_pos_pair_yx, [K,1])
      shape_xy = tf.shape(aff_xy)
      
      top_k_pos_pair_xy = tf.reshape(top_k_pos_pair_xy, [shape_xy[0],K,-1])
      top_k_pos_pair_yx = tf.reshape(top_k_pos_pair_yx, [-1,K,shape_xy[1]])

      loss_xy = tf.maximum(0.,self.opts.margin + top_k_pos_pair_xy - tf.expand_dims(top_k_neg_pair_xy,2))
      loss_yx = tf.maximum(0.,self.opts.margin + top_k_pos_pair_yx - tf.expand_dims(top_k_neg_pair_yx,0))

      loss_xy = tf.reduce_mean(tf.reshape(loss_xy,[-1]))
      loss_yx = tf.reduce_mean(tf.reshape(loss_yx,[-1]))

      # xy_nonzero = tf.reduce_sum(tf.cast(tf.greater(loss_xy, 0.), tf.float32)) + 1e-13
      # yx_nonzero = tf.reduce_sum(tf.cast(tf.greater(loss_yx, 0.), tf.float32)) + 1e-13
      #
      # loss_xy = loss_xy / xy_nonzero
      # loss_yx = loss_yx / yx_nonzero

      return loss_xy, loss_yx
      