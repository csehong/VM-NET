import tensorflow as tf
import numpy as np

import OPTS

class Recall:
  class OPTS(OPTS.OPTS):
    class DIST:
      L2 = 0
      COS = 1
    def __init__(self):
      OPTS.OPTS.__init__(self,'Recall OPTS')
      self.network_name = None
      self.distance = self.DIST.COS
    
  def __init__(self, opts=None):
    if opts is None:
      opts = self.OPTS()
    self.opts = opts
    self.opts.assert_all_keys_valid()
    
  def construct(self, embed_x, embed_y, aff_xy, K_list):
    embed_x = tf.nn.l2_normalize(embed_x, 1)
    embed_y = tf.nn.l2_normalize(embed_y, 1)
        
    dist_xy = tf.matmul(embed_x, embed_y,transpose_a = False, transpose_b = True)
    
    self.to_debug = []
    
    shape_xy = tf.shape(aff_xy)
    recall_k_xy = []
    recall_k_yx = []
    recall_k_xy_idx = 0
    recall_k_yx_idx = 0
    for K in K_list:
      indice_mat_xy = tf.tile(tf.expand_dims(tf.range(0,shape_xy[0]),1),[1,K])
      indice_mat_xy = tf.expand_dims(indice_mat_xy, dim=1)
      indice_mat_yx = tf.tile(tf.expand_dims(tf.range(0,shape_xy[1]),1),[1,K])
      indice_mat_yx = tf.expand_dims(indice_mat_yx, dim=1)
      
      _, xy_idx = tf.nn.top_k(dist_xy,k=K)
      _, yx_idx = tf.nn.top_k(tf.transpose(dist_xy),k=K)

      # variable for recall result
      recall_k_xy_idx = xy_idx
      recall_k_yx_idx = yx_idx

      xy_idx = tf.expand_dims(xy_idx, dim=1)
      yx_idx = tf.expand_dims(yx_idx, dim=1)
      xy_idx = tf.concat(1, [indice_mat_xy,xy_idx])
      yx_idx = tf.concat(1, [yx_idx,indice_mat_yx])

      xy_idx = tf.transpose(xy_idx, perm=[0,2,1])
      yx_idx = tf.transpose(yx_idx, perm=[0,2,1])

      xy_search = tf.cast(tf.gather_nd(aff_xy, xy_idx),dtype=tf.float32)
      yx_search = tf.cast(tf.gather_nd(aff_xy, yx_idx),dtype=tf.float32)

      xy_result = tf.cast(tf.not_equal(tf.reduce_sum(xy_search,1), 0.),tf.float32)
      yx_result = tf.cast(tf.not_equal(tf.reduce_sum(yx_search,1), 0.),tf.float32)

      recall_k_xy.append(tf.reduce_mean(xy_result))
      recall_k_yx.append(tf.reduce_mean(yx_result))

    return recall_k_xy, recall_k_yx, recall_k_xy_idx, recall_k_yx_idx
  
  
if __name__ == '__main__':
  rec_opts = Recall.OPTS()
  rec_opts.network_name = 'Test_Object'
  
  recall = Recall(rec_opts)
  
  embed_x = np.random.normal(size=[20,10])
  embed_y = np.random.normal(size=[20,10])
  aff_xy = np.eye(20,20)
  K_list = [3]
  r_xy, r_yx = recall.construct(embed_x, embed_y, aff_xy, K_list)
  import scipy.io
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    d, r1, r2 = sess.run([recall.to_debug, r_xy, r_yx])
    print r1, r2
    scipy.io.savemat('recall_test.mat',mdict={'d':d})
  
  