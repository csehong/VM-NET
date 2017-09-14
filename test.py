#!/usr/bin/python
import tensorflow as tf
import numpy as np
import os
import scipy.io

from network import Model

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('num_layer_x', 3, 'Constraint Weight xy')
flags.DEFINE_integer('num_layer_y',2, 'Constraint Weight yx')
flags.DEFINE_integer('test_batch_size', 1086, 'Test batch size.') #flags.DEFINE_integer('test_batch_size', 1000, 'Test batch size.')
flags.DEFINE_string('summaries_dir', 'expr_ware/lr_0.0003   dr_0.9   nx_3   ny_2   xy_3   yx_1   K_100   ba_2000', 'Directory to put the summary and log data.')

net_opts = Model.OPTS()
net_opts.network_name = 'Wrapping Network'
net_opts.x_dim = 1140 #4096
net_opts.y_dim = 1024 #6000
net_opts.x_num_layer = FLAGS.num_layer_x
net_opts.y_num_layer = FLAGS.num_layer_y
net = Model(net_opts)
net.construct()

saver = tf.train.Saver(tf.global_variables())
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  checkpoint_dir = os.path.join(FLAGS.summaries_dir, 'checkpoints')
  checkpoint_prefix = os.path.join(checkpoint_dir, "model.ckpt")
  if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

  step = 0
  ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
  if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    #saver.restore(sess, os.path.join(checkpoint_dir, 'model.ckpt-9399'))
    step = int(ckpt.model_checkpoint_path.split('-')[-1])
    step += 1
    print('Session restored successfully. step: {0}'.format(step))

  import data.himv as himv
  num_test = FLAGS.test_batch_size
  data_manager = himv.Data_Manager()
  data_name = data_manager.data_name
  data_name_test = data_name[-num_test:len(data_name)]
  test_batch = data_manager.batch_iterator_thread(FLAGS.test_batch_size, data_manager.validation_pair_list, is_train=False)
  # test_batch = data_manager.batch_iterator_thread(FLAGS.test_batch_size, data_manager.test_pair_list, is_train=False)

  K = 100
  x_batch, y_batch, aff_xy = test_batch.next()
  xy, yx, xy_idx, yx_idx = sess.run([net.recall_xy, net.recall_yx, net.xy_idx, net.yx_idx], feed_dict={
      net.x_data: x_batch,
      net.y_data: y_batch,
      net.K: K,
      net.aff_xy: aff_xy,
      net.keep_prob: 1., net.is_training: False})  # actually, False


  print("[iter %d] xy: %s, yx: %s, " % (step, xy, yx))



  import csv
  with open('./recall_xy(music-video)_general_vali.csv', 'wb') as csvfile:
      writer = csv.writer(csvfile, delimiter=',')
      top = ["Query Name"] + [str(i) for i in range(0, num_test)]
      writer.writerow(top)
      for idx_input in range(len(xy_idx)):
          recall_mv = [data_name_test[idx_recall]  for idx_recall in xy_idx[idx_input]]
          result = [data_name_test[idx_input]] + recall_mv
          writer.writerow(result)

  with open('./recall_yx(video-music)_general_vali.csv', 'wb') as csvfile:
      writer = csv.writer(csvfile, delimiter=',')
      top = ["Query Name"] + [str(i) for i in range(0, num_test)]
      writer.writerow(top)
      for idx_input in range(len(yx_idx)):
          recall_mv = [data_name_test[idx_recall] for idx_recall in yx_idx[idx_input]]
          result = [data_name_test[idx_input]] + recall_mv
          writer.writerow(result)

