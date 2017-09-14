#!/usr/bin/python

import tensorflow as tf
import numpy as np
import random
import OPTS




class Triplet_Structure:
  class OPTS(OPTS.OPTS):
    class DIST:
      L2 = 0
      COS = 1
      
    def __init__(self):
      OPTS.OPTS.__init__(self,'Triplet Structure_OPTS')
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
  
  def construct(self, low_feature, embed_feature, K, name):
    with tf.variable_scope(name):

      low_feature = tf.nn.l2_normalize(low_feature, 1)


      # convert simliarity to distance by (-1) >> high value indicates the distance between the samples is long
      dist_low = (-1) * tf.matmul(low_feature, low_feature, transpose_a=False, transpose_b=True)
      dist_embed = (-1) * tf.matmul(embed_feature, embed_feature, transpose_a=False, transpose_b=True)

      # d1_low = tf.slice(dist_low, [2, 0], [K, 1])
      # d2_low = tf.slice(dist_low, [2, 1], [K, 1])
      # d1_embed = tf.slice(dist_embed, [2, 0], [K, 1])
      # d2_embed = tf.slice(dist_embed, [2, 1], [K, 1])
      # coeff = tf.sign(d1_low - d2_low) - tf.sign(d1_embed - d2_embed)
      # dist_inverse = (d2_embed - d1_embed)    # -1 * (d1_embed - d2_embed)
      # dist_tensor = tf.multiply(coeff, dist_inverse)
      # loss_structure = tf.reduce_mean(dist_tensor)

      d1_low = tf.slice(dist_low, [0, 0], [K, K-1])
      d2_low = tf.slice(dist_low, [0, 1], [K, K-1])
      d1_embed = tf.slice(dist_embed, [0, 0], [K, K-1])
      d2_embed = tf.slice(dist_embed, [0, 1], [K, K-1])
      coeff = tf.sign(d1_low - d2_low) - tf.sign(d1_embed - d2_embed)
      dist_inverse = (d2_embed - d1_embed)  # -1 * (d1_embed - d2_embed)
      dist_tensor = tf.multiply(coeff, dist_inverse)
      loss_structure = tf.reduce_mean(dist_tensor)


      return loss_structure
