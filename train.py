#!/usr/bin/python

import tensorflow as tf
import numpy as np
import os
import scipy.io

from network import Model
from network_structure import Model_structure
from Logger import Logger

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 3e-4
                   , 'Initial learning rate.')
flags.DEFINE_float('keep_prob', 0.9, 'Dropout keep probability rate.')
flags.DEFINE_integer('num_layer_x', 3, 'Constraint Weight xy')
flags.DEFINE_integer('num_layer_y',2, 'Constraint Weight yx')
flags.DEFINE_integer('constraint_xy', 3, 'Constraint Weight xy')
flags.DEFINE_integer('constraint_yx',1, 'Constraint Weight yx')
flags.DEFINE_integer('constraint_x', 0.2, 'Constraint Structure Weight x')
flags.DEFINE_integer('constraint_y', 0.2, 'Constraint Structure Weight y')
flags.DEFINE_integer('top_K', 100, 'Top mosk K number for violation')
flags.DEFINE_float('weight_decay', 0, 'Weight decay.')
flags.DEFINE_integer('max_steps', 700000, 'Number of steps to run trainer.')
flags.DEFINE_integer('train_batch_size', 1000, 'Train Batch size.')  #flags.DEFINE_integer('batch_size', 1500, 'Batch size.')
flags.DEFINE_integer('validation_batch_size', 4000, 'Validation Batch size.')  #flags.DEFINE_integer('batch_size', 1500, 'Batch size.')
flags.DEFINE_integer('test_batch_size', 1000, 'Test batch size.') #flags.DEFINE_integer('test_batch_size', 1000, 'Test batch size.')
flags.DEFINE_integer('display_step', 50, 'Train display step.')
flags.DEFINE_integer('test_step', 200, 'Test step.')
flags.DEFINE_integer('save_step', 200, 'Checkpoint saving step.')
flags.DEFINE_string('summaries_dir', "dir_name", 'Directory to put the summary and log data.')
FLAGS.summaries_dir = "expr/Structure_test(K2_Linear)_" + "lr_" + str(FLAGS.learning_rate) + "   dr_" + str(FLAGS.keep_prob) +  "   nx_" + str(FLAGS.num_layer_x) +   "   ny_" + str(FLAGS.num_layer_y)  \
                    +  "   xy_" + str(FLAGS.constraint_xy) +"   yx_" + str(FLAGS.constraint_yx) + "   x_" + str(FLAGS.constraint_x) +"   y_" + str(FLAGS.constraint_y) \
                    + "   K_" + str(FLAGS.top_K) + "   ba_" + str(FLAGS.train_batch_size)



# Structure_test(K2_dom)_lr_0.0003   dr_0.7   nx_3   ny_2   xy_3   yx_1   x_0.2   y_0.2   K_100   ba_1000

# net_opts = Model.OPTS()
# net_opts.network_name = 'Wrapping Network'
# net_opts.x_dim = 1140
# net_opts.y_dim = 1024
# net_opts.x_num_layer = FLAGS.num_layer_x
# net_opts.y_num_layer = FLAGS.num_layer_y
# net_opts.constraint_weights = [FLAGS.constraint_xy, FLAGS.constraint_yx]
# net_opts.is_linear = False
# net = Model(net_opts)
# net.construct()

# net_opts = Model.OPTS()
net_opts = Model_structure.OPTS()
net_opts.network_name = 'Wrapping Network'
net_opts.x_dim = 1140
net_opts.y_dim = 1024
net_opts.x_num_layer = FLAGS.num_layer_x
net_opts.y_num_layer = FLAGS.num_layer_y
net_opts.constraint_weights = [FLAGS.constraint_xy, FLAGS.constraint_yx, FLAGS.constraint_x, FLAGS.constraint_y]
net_opts.is_linear = True
# net = Model(net_opts)
net = Model_structure(net_opts)
net.construct()


lr = tf.placeholder(tf.float32, name='learning_rate')
loss = net.loss
# loss = net.loss + net.dom_loss


global_step_ = tf.Variable(0, trainable=False)
decay_step = 1000
# learning_rate_ = tf.train.exponential_decay(FLAGS.learning_rate, global_step_,
#                                            decay_step, 0.96, staircase=True)
learning_rate = FLAGS.learning_rate

# top_k_ = tf.train.exponential_decay(float(FLAGS.top_K), global_step_,
#                                            decay_step, 0.96, staircase=True)
top_k = FLAGS.top_K


update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
# Ensures that we execute the update_ops before performing the train_step (ref: http://ruishu.io/2016/12/27/batchnorm/ )
with tf.control_dependencies(update_ops):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    # optimizer = tf.train.AdamOptimizer(learning_rate_).minimize(loss, global_step=global_step_)

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
saver = tf.train.Saver(tf.all_variables(), max_to_keep = None)
with tf.Session(config=config) as sess:
  sess.run(tf.global_variables_initializer())
  
  train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/train',sess.graph)
  validation_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/validation')
  test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/test')
  
  checkpoint_dir = os.path.join(FLAGS.summaries_dir, 'checkpoints')
  checkpoint_prefix = os.path.join(checkpoint_dir, "model.ckpt")
  if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
  
  logger = Logger(FLAGS.summaries_dir)
  logger.write(str(FLAGS.__flags))
  import data.himv as himv
  data_manager = himv.Data_Manager()
  train_batch = data_manager.batch_iterator_thread(FLAGS.train_batch_size, data_manager.train_pair_list)
  validation_batch = data_manager.batch_iterator_thread(FLAGS.validation_batch_size, data_manager.validation_pair_list, is_train=False)
  test_batch = data_manager.batch_iterator_thread(FLAGS.test_batch_size, data_manager.test_pair_list, is_train=False)
  
  step = 0
  ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
  if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    step = int(ckpt.model_checkpoint_path.split('-')[-1])
    print('Session restored successfully. step: {0}'.format(step))
    step = step + 1

  max_step_acc_xy = [0, -5e3]
  max_step_acc_yx = [0, -5e3]
  for i in range(step,FLAGS.max_steps):
    p = float(i) / FLAGS.max_steps
    lamb = 0
    # lamb = 2. / (1. + np.exp(-20. * p)) - 1   # lamb = 2. / (1. + np.exp(-10. * p)) - 1
    # lamb *= 10

    x_batch, y_batch, aff_xy = train_batch.next()
    # K = FLAGS.top_K

    # learning_rate = sess.run(learning_rate_)





    sess.run(optimizer,feed_dict={
                      net.x_data:x_batch,
                      net.y_data:y_batch,
                      net.K:int(top_k),
                      net.aff_xy:aff_xy,
                      net.keep_prob:FLAGS.keep_prob,
                      lr: learning_rate, net.is_training: True, net.l: lamb})
    
    if (i+1) % FLAGS.display_step == 0:
      loss_cross_xy, loss_single_x, loss_cross_yx, loss_single_y, xy_rank_x, yx_rank_x, x_rank_x, xy_rank_y, yx_rank_y, y_rank_y, l,xy,yx,dl,da = sess.run([net.loss_cross_xy, net.loss_single_x, net.loss_cross_yx, net.loss_single_y, net.xy_rank_x, net.yx_rank_x, net.x_rank_x, net.xy_rank_y, net.yx_rank_y, net.y_rank_y, loss,net.recall_xy, net.recall_yx, net.dom_loss, net.dom_acc],feed_dict={
                      net.x_data:x_batch,
                      net.y_data:y_batch,
                      net.K:int(top_k),
                      net.aff_xy:aff_xy,
                      net.keep_prob:1., net.is_training: True, net.l: lamb})


      logger.write("[iter %d] (%s %s, %s %s)(%s %s, %s) (%s %s, %s) loss=%.4g, %s, %s "%(i+1, loss_cross_xy, loss_cross_yx, loss_single_x, loss_single_y,  xy_rank_x, yx_rank_x, x_rank_x, xy_rank_y, yx_rank_y, y_rank_y, l, xy, yx))
      short_summary = tf.Summary(value=[
        tf.Summary.Value(tag="triplet_loss", simple_value=float(l)),
        tf.Summary.Value(tag="K", simple_value=float(top_k)),
        tf.Summary.Value(tag="lambda", simple_value=float(lamb)),
        tf.Summary.Value(tag="dom_loss", simple_value=float(dl)),
        tf.Summary.Value(tag="dom_acc", simple_value=float(da)),
        tf.Summary.Value(tag="learning_rate", simple_value=float(learning_rate)),
        tf.Summary.Value(tag="recall/xy_R@1", simple_value=float(xy[0])),
        tf.Summary.Value(tag="recall/xy_R@10", simple_value=float(xy[1])),
        tf.Summary.Value(tag="recall/xy_R@25", simple_value=float(xy[2])),
        tf.Summary.Value(tag="recall/xy_R@50", simple_value=float(xy[3])),
        tf.Summary.Value(tag="recall/xy_R@100", simple_value=float(xy[4])),
        tf.Summary.Value(tag="recall/xy_R@1000", simple_value=float(xy[5])),
        tf.Summary.Value(tag="recall/yx_R@1", simple_value=float(yx[0])),
        tf.Summary.Value(tag="recall/yx_R@10", simple_value=float(yx[1])),
        tf.Summary.Value(tag="recall/yx_R@25", simple_value=float(yx[2])),
        tf.Summary.Value(tag="recall/yx_R@50", simple_value=float(yx[3])),
        tf.Summary.Value(tag="recall/yx_R@100", simple_value=float(yx[4])),
        tf.Summary.Value(tag="recall/yx_R@1000", simple_value=float(yx[5])),
      ])
      train_writer.add_summary(short_summary, i)
      #scipy.io.savemat('debug.mat', mdict={'d':dl})
      #exit()


    if (i+1) % FLAGS.test_step == 0:
      # # for validation
      # x_batch, y_batch, aff_xy = validation_batch.next()
      # l, xy, yx, xy_idx, yx_idx, dl, da = sess.run(
      #   [loss, net.recall_xy, net.recall_yx, net.xy_idx, net.yx_idx, net.dom_loss, net.dom_acc], feed_dict={
      #     net.x_data: x_batch,
      #     net.y_data: y_batch,
      #     net.K: int(top_k),
      #     net.aff_xy: aff_xy,
      #     net.keep_prob: 1., net.is_training: False, net.l: lamb})  # actually, False
      #
      # logger.write("[Validation iter %d] loss=%.4g, %s, %s" % (i + 1, l, xy, yx))
      # short_summary = tf.Summary(value=[
      #   tf.Summary.Value(tag="triplet_loss", simple_value=float(l)),
      #   tf.Summary.Value(tag="K", simple_value=float(top_k)),
      #   tf.Summary.Value(tag="lambda", simple_value=float(lamb)),
      #   tf.Summary.Value(tag="dom_loss", simple_value=float(dl)),
      #   tf.Summary.Value(tag="dom_acc", simple_value=float(da)),
      #   tf.Summary.Value(tag="learning_rate", simple_value=float(learning_rate)),
      #   tf.Summary.Value(tag="recall/xy_R@1", simple_value=float(xy[0])),
      #   tf.Summary.Value(tag="recall/xy_R@10", simple_value=float(xy[1])),
      #   tf.Summary.Value(tag="recall/xy_R@25", simple_value=float(xy[2])),
      #   tf.Summary.Value(tag="recall/xy_R@50", simple_value=float(xy[3])),
      #   tf.Summary.Value(tag="recall/xy_R@100", simple_value=float(xy[4])),
      #   tf.Summary.Value(tag="recall/xy_R@1000", simple_value=float(xy[5])),
      #   tf.Summary.Value(tag="recall/yx_R@1", simple_value=float(yx[0])),
      #   tf.Summary.Value(tag="recall/yx_R@10", simple_value=float(yx[1])),
      #   tf.Summary.Value(tag="recall/yx_R@25", simple_value=float(yx[2])),
      #   tf.Summary.Value(tag="recall/yx_R@50", simple_value=float(yx[3])),
      #   tf.Summary.Value(tag="recall/yx_R@100", simple_value=float(yx[4])),
      #   tf.Summary.Value(tag="recall/yx_R@1000", simple_value=float(yx[5])),
      # ])
      # validation_writer.add_summary(short_summary, i)

      # for test
      x_batch, y_batch, aff_xy = test_batch.next()
      l,xy,yx, xy_idx, yx_idx,dl,da = sess.run([loss,net.recall_xy, net.recall_yx, net.xy_idx, net.yx_idx, net.dom_loss, net.dom_acc],feed_dict={
                      net.x_data:x_batch,
                      net.y_data:y_batch,
                      net.K:int(top_k),  # caution: We currently use topK as test batch size
                      net.aff_xy:aff_xy,
                      net.keep_prob:1., net.is_training: False, net.l: lamb})  #actually, False

      logger.write("[TEST iter %d] loss=%.4g, %s, %s"%(i+1, l, xy, yx))
      short_summary = tf.Summary(value=[
        tf.Summary.Value(tag="triplet_loss", simple_value=float(l)),
        tf.Summary.Value(tag="K", simple_value=float(top_k)),
        tf.Summary.Value(tag="lambda", simple_value=float(lamb)),
        tf.Summary.Value(tag="dom_loss", simple_value=float(dl)),
        tf.Summary.Value(tag="dom_acc", simple_value=float(da)),
        tf.Summary.Value(tag="learning_rate", simple_value=float(learning_rate)),
        tf.Summary.Value(tag="recall/xy_R@1", simple_value=float(xy[0])),
        tf.Summary.Value(tag="recall/xy_R@10", simple_value=float(xy[1])),
        tf.Summary.Value(tag="recall/xy_R@25", simple_value=float(xy[2])),
        tf.Summary.Value(tag="recall/xy_R@50", simple_value=float(xy[3])),
        tf.Summary.Value(tag="recall/xy_R@100", simple_value=float(xy[4])),
        tf.Summary.Value(tag="recall/xy_R@1000", simple_value=float(xy[5])),
        tf.Summary.Value(tag="recall/yx_R@1", simple_value=float(yx[0])),
        tf.Summary.Value(tag="recall/yx_R@10", simple_value=float(yx[1])),
        tf.Summary.Value(tag="recall/yx_R@25", simple_value=float(yx[2])),
        tf.Summary.Value(tag="recall/yx_R@50", simple_value=float(yx[3])),
        tf.Summary.Value(tag="recall/yx_R@100", simple_value=float(yx[4])),
        tf.Summary.Value(tag="recall/yx_R@1000", simple_value=float(yx[5])),
      ])

      # if (float(xy[0]) > max_step_acc_xy[1]):
      #   max_step_acc_xy[1] = float(xy[0])
      #   max_step_acc_xy[0] = i + 1
      # if (float(yx[0]) > max_step_acc_yx[1]):
      #   max_step_acc_yx[1] = float(yx[0])
      #   max_step_acc_yx[0] = i + 1

      test_writer.add_summary(short_summary, i)
    if (i+1) % FLAGS.save_step == 0:
      saver.save(sess,checkpoint_prefix,global_step=i)
      logger.write("[Checkpoint at step %d saved]"%(i+1))
      logger.write(FLAGS.summaries_dir)
      # logger.write("*** xy (%d, %f), yx (%d, %f) ***" %(max_step_acc_xy[0], max_step_acc_xy[1], max_step_acc_yx[0], max_step_acc_yx[1]))

  
  
      