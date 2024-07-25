import sys
import os.path
sys.path.insert(0, os.path.abspath("./simple-dnn"))

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy
import time
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
import os
import statistics
import tensorflow.contrib.losses as losses
from simple_dnn.util.format import UnitPosNegScale, reshape_pad
from triplet import triplet_center_loss, center_loss
from sklearn.cluster import KMeans

class OpenNetBase(object):
    """ OpenNet base class.
    """
    def __init__(self, x_dim, y_dim,
                 z_dim=6,
                 x_scale=UnitPosNegScale.scale,
                 x_inverse_scale=UnitPosNegScale.inverse_scale,
                 x_reshape=None,
                 opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9),
                 recon_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9),
                 c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9),
                 rot_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9),

                 dist='mean_separation_spread',
                 decision_dist_fn = 'euclidean',
                 threshold_type='perclass', #global
                 dropout = True, keep_prob=0.7,
                 batch_size=128, iterations=5000,
                 display_step=500, save_step=500,
                 model_directory=None,  # Directory to save trained model to.
                 density_estimation_factory=None, # Depricated
                 load_model_directory=None,
                 ce_loss=True, recon_loss=False, inter_loss=True, intra_loss=True,
                 tc_loss=False,
                 div_loss=False, combined_loss=False,
                 decouple_loss=False,
                 trans_ce_loss = False,
                 intra_sample_loss=False,
                 cluster_loss=False,
                 mmf_extension=False,
                 contamination=0.01,
                 mmf_comb = 0.0,
                 margin = 8.0, self_sup = False, n_rots=0, n_decouples=0, n_clusters=2
                 ):
        """
        Args:
        :param x_dim - dimension of the input
        :param y_dim - number of known classes.
        :param z_dim - the number of latent variables.
        :param x_scale - an input scaling function. Default scale to range of [-1, 1].
                         If none, the input will not be scaled.
        :param x_inverse_scale - reverse scaling fn. by rescaling from [-1, 1] to original input scale.
                                 If None, the the output of decoder(if there is a decoder) will rescaled.
        :param x_reshape - a function to reshape the input before feeding to the networks input layer.
                            If None, the input will not be reshaped.
        :param opt - the Optimizer used when updating based on ii-loss.
                     Used when inter_loss and intra_loss are enabled. Default is AdamOptimizer.
        :param recon_opt - the Optimizer used when updating based on reconstruction-loss (Not used ii, ii+ce or ce).
                           Used when recon_loss is enabled. Default is AdamOptimizer.
        :param c_opt - the Optimizer used when updating based on cross entropy loss.
                       Used for ce and ii+ce modes (i.e. ce_loss is enabled). Default is AdamOptimizer.
        :param batch_size - training barch size.
        :param iterations - number of training iterations.
        :param display_step - training info displaying interval.
        :param save_step - model saving interval.
        :param model_directory - derectory to save model in.
        :param dist - ii-loss calculation mode. Only 'mean_separation_spread' should be used.
        :param decision_dist_fn - outlier score distance functions
        :param threshold_type - outlier threshold mode. 'global' appears to give better results.
        :param ce_loss - Consider cross entropy loss. When enabled with intra_loss and inter_loss gives (ii+ce) mode.
        :param recon_loss - Experimental! avoid enabling them.
        :param inter_loss - Consider inter-class separation. Should be enabled together with intra_loss for (ii-loss).
        :param intra_loss - Consider intra-class spread. Should be enabled together with inter_loss for (ii-loss).
        :param div_loss and combined_loss - Experimental. avoid enabling them.
        :param contamination - contamination ratio used for outlier threshold estimation.
        """
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.n_clusters = n_clusters

        self.x_scale = x_scale
        self.x_inverse_scale = x_inverse_scale
        self.x_reshape = x_reshape

        self.z_dim = z_dim
        self.rot_dim = n_rots
        self.decouple_dim = n_decouples

        self.dropout = dropout
        self.is_training = False
        self.keep_prob = keep_prob

        self.contamination = contamination

        self.opt = opt
        self.recon_opt = recon_opt
        self.c_opt = c_opt
        self.rot_opt = rot_opt

        assert dist in ['class_mean', 'all_pair', 'mean_separation_spread', 'min_max', 'triplet_center_loss','others']
        self.dist = dist

        self.decision_dist_fn = decision_dist_fn

        assert threshold_type in ['global', 'perclass']
        self.threshold_type = threshold_type
        self.comb = mmf_comb
        self.margin = margin
        # Training Config
        self.batch_size = batch_size
        self.iterations = iterations
        self.display_step = display_step
        self.save_step = save_step
        self.model_directory = model_directory
        self.load_model_directory = load_model_directory

        self.enable_trans_ce_loss, self.enable_ce_loss, self.enable_recon_loss, \
            self.enable_inter_loss, self.enable_intra_loss, self.div_loss, self.enable_mmf_extension, self.enable_tc_loss, \
        self.enable_self, self.enable_decouple, self.enable_intra_sample_loss, self.enable_cluster_loss = \
                trans_ce_loss, ce_loss, recon_loss, inter_loss, intra_loss, div_loss, mmf_extension, tc_loss, self_sup, decouple_loss, intra_sample_loss, cluster_loss

        self.graph = tf.Graph()

        self.model_params = ['x_dim', 'y_dim', 'z_dim', 'dropout', 'keep_prob',
                             'contamination', 'decision_dist_fn', 'dist', 'batch_size',
                             'batch_size', 'iterations', 'enable_ce_loss',
                             'enable_recon_loss', 'enable_inter_loss', 'enable_intra_loss', 'enable_tc_loss',
                             'div_loss', 'enable_mmf_extension', 'threshold_type']

        with self.graph.as_default():
            self.sess = tf.Session()
      #      print("build model ing...")
            self.build_model()

            # To save and restore all the variables.
            self.saver = tf.train.Saver()

    def save_model(self, model_file_name):
      if self.model_directory is None:
        return 'ERROR: Model directory is None'
      if not os.path.exists(self.model_directory):
          os.makedirs(self.model_directory)
      return self.saver.save(self.sess, os.path.join(self.model_directory, model_file_name))

    def model_config(self):
        return {field: val for field, val in vars(self).items() if field in self.model_params}

    def x_reformat(self, xs):
      """ Rescale and reshape x if x_scale and x_reshape functions are provided.
      """
      if self.x_scale is not None:
        xs = self.x_scale(xs)
      if self.x_reshape is not None:
        xs = self.x_reshape(xs)
      return xs

    def _next_batch(self, x, y, rot=None, decouple=None):
        index = np.random.randint(0, high=x.shape[0], size=self.batch_size)
        if not (rot is None) and not (decouple is None):
            return x[index], y[index], rot[index], decouple[index]
        elif decouple is None and not (rot is None):
            return x[index], y[index], rot[index]
        else:
            return  x[index], y[index]


    def encoder(self, x, reuse=False):
        """ Encoder network.
        Args:
            :param x - input x.
            :param reuse - whether to reuse old network on create new one.
        Returns:
            latent var z
        """
        pass

    def decoder(self, z, reuse=False):
        """ Decoder Network. Experimental!
        Args:
            :param z - latent variables z.
            :param reuse - whether to reuse old network on create new one.
        Returns:
            The reconstructed x
        """
        pass

    def bucket_mean(self, data, bucket_ids, num_buckets):
        total = tf.unsorted_segment_sum(data, bucket_ids, num_buckets)
        count = tf.unsorted_segment_sum(tf.ones_like(data), bucket_ids, num_buckets)
        return total / count

    def bucket_max(self, data, bucket_ids, num_buckets):
        b_max = tf.unsorted_segment_max(data, bucket_ids, num_buckets)
        return b_max

    def sq_difference_from_mean(self, data, class_mean, dim):
        """ Calculates the squared difference from clas mean.
        """
        sq_diff_list = []
        for i in range(dim):
            sq_diff_list.append(tf.reduce_mean(
                tf.squared_difference(data, class_mean[i]), axis=1))

        return tf.stack(sq_diff_list, axis=1)

    def sq_difference_from_zero(self, data):
        """ Calculates the squared difference from clas mean.
        """
        dist = tf.math.reduce_euclidean_norm(data, 1)
        return tf.reduce_min(dist)

    def chunks(self, lst):
        """Yield successive n-sized chunks from lst."""
        ans = []
        for i in range(0, self.rot_dim*4, self.rot_dim):
            ans.append(lst[i:i + self.rot_dim])
        return ans

    def intra_sample_dist(self, data, class_means):
        data_orig = data[0::2]
        data_rot = data[1::2]
        sq_diff = tf.reduce_sum(tf.squared_difference(class_means, data_rot), axis=1)
        return tf.reduce_mean(sq_diff)

    def intra_sample_distance2(self, data):
        data1 = data[0::2]
        data2 = data[1::2]
        chuncks = tf.stack([data1, data2])
        centroids = tf.reduce_mean(chuncks, axis=0)
        print("centroids shape")
        print(centroids.shape)
        chunck_diff_1 = tf.reduce_sum(tf.squared_difference(data1, centroids), axis=1)
        chunck_diff_2 = tf.reduce_sum(tf.squared_difference(data2, centroids), axis=1)
        chunck_diff_all = tf.concat([chunck_diff_1, chunck_diff_2], 0)
        print("diff shape")
        print(chunck_diff_all.shape)
        return tf.reduce_mean(chunck_diff_all)

    def normalize(self, output):
        return (output - tf.reduce_mean(output, axis=0)) / tf.math.reduce_std(
            output, axis=0
        )

    def get_off_diag(self, c):
        zero_diag = tf.zeros(c.shape[-1])
        return tf.linalg.set_diag(c, zero_diag)

    def cross_corr_matrix(self, z_a_norm, z_b_norm):
        return tf.linalg.matmul(tf.transpose(z_a_norm),z_b_norm)

    def intra_sample_distance3(self, data):
        data_split = tf.split(data, 4)
        data1 = data_split[0]
        data2 = data_split[1]
        data3 = data_split[2]
        data4 = data_split[3]
        data1_norm, data2_norm, data3_norm, data4_norm = self.normalize(data1), self.normalize(data2), self.normalize(data3), self.normalize(data4)
        c2, c3, c4 = self.cross_corr_matrix(data1_norm, data2_norm), self.cross_corr_matrix(data1_norm, data3_norm), self.cross_corr_matrix(data1_norm, data4_norm)
        c2_diff = tf.pow(tf.linalg.diag_part(c2) - 1, 2)
        c3_diff = tf.pow(tf.linalg.diag_part(c3) - 1, 2)
        c4_diff = tf.pow(tf.linalg.diag_part(c4) - 1, 2)
        off2_diag = tf.pow(self.get_off_diag(c2), 2)
        off3_diag = tf.pow(self.get_off_diag(c3), 2)
        off4_diag = tf.pow(self.get_off_diag(c4), 2)
        loss = tf.reduce_sum(tf.stack([c2_diff, c3_diff, c4_diff])) + tf.reduce_sum(tf.stack([off2_diag, off3_diag, off4_diag]))
        return loss

    def intra_sample_distance(self, data):
       # print("data shape ", data.shape())
        data_split = tf.split(data, 4)
        data1 = data_split[0]
        data2 = data_split[1]
        data3 = data_split[2]
        data4 = data_split[3]
        chuncks = tf.stack([data1, data2, data3, data4])
     #   centroids = tf.reduce_mean(chuncks, axis=0)
        # dim = data1.shape()[0]
        # print("dim ", dim)
        # ap_dist = self.all_pair_distance(centroids)
        # not_diag_mask = tf.logical_not(tf.cast(tf.eye(dim), dtype=tf.bool))
        # inter_separation = tf.reduce_min(tf.boolean_mask(tensor=ap_dist, mask=not_diag_mask))
      #  chunck_diff_1 = tf.reduce_sum(tf.squared_difference(data1, centroids), axis=1)
        chunck_diff_2 = tf.reduce_sum(tf.squared_difference(data2, data1), axis=1)
        chunck_diff_3 = tf.reduce_sum(tf.squared_difference(data3, data1), axis=1)
        chunck_diff_4 = tf.reduce_sum(tf.squared_difference(data4, data1), axis=1)
        chunck_diff_all = tf.concat([chunck_diff_2, chunck_diff_3, chunck_diff_4], 0)
        print("diff shape")
        print(chunck_diff_all.shape)
        return tf.reduce_mean(chunck_diff_all)

    def intra_sample_dist2(self, data, labels, class_mean):
        sq_diff_list = []
        # print('till here')
        # print("rot_dim shape")
        # print(self.rot_dim)
        # print(class_mean.shape)
        # print(data.shape)
        for i in range(self.rot_dim):
        #    print("dim i" + str(i))
            sq_diff_list.append(tf.reduce_mean(tf.squared_difference(data, class_mean[i]), axis=1))
        sq_diff = tf.stack(sq_diff_list, axis=1)
        inter_intra_sq_diff = self.bucket_mean(sq_diff, labels, 2)
        intra_class_sq_diff = inter_intra_sq_diff[1]
        return intra_class_sq_diff

    def inter_min_intra_max(self, data, labels, class_mean, dim):
        """ Calculates intra-class spread as max distance from class means.
        Calculates inter-class separation as the distance between the two closest class means.
        """
        _, inter_min = self.inter_separation_intra_spred(data, labels, class_mean, dim)
        sq_diff = self.sq_difference_from_mean(data, class_mean, dim)
        # Do element wise mul with labels to use as mask
        masked_sq_diff = tf.multiply(sq_diff, tf.cast(labels, dtype=tf.float32))
        intra_max = tf.reduce_sum(tf.reduce_max(masked_sq_diff, axis=0))

        return intra_max, inter_min

    def inter_intra_diff(self, data, labels, class_mean, dim):
        """ Calculates the intra-class and inter-class distance
        as the average distance from the class means.
        """
        sq_diff = self.sq_difference_from_mean(data, class_mean, dim)
        inter_intra_sq_diff = self.bucket_mean(sq_diff, labels, 2)
        inter_class_sq_diff = inter_intra_sq_diff[0]
        intra_class_sq_diff = inter_intra_sq_diff[1]
        return intra_class_sq_diff, inter_class_sq_diff

    def inter_separation_intra_spred(self, data, labels, class_mean, dim):
        """ Calculates intra-class spread as average distance from class means.
        Calculates inter-class separation as the distance between the two closest class means.
        Returns:
        intra-class spread and inter-class separation.
        """
        intra_class_sq_diff, _ = self.inter_intra_diff(data, labels, class_mean, dim)
        ap_dist = self.all_pair_distance(class_mean)
        dim = tf.shape(class_mean)[0]
        not_diag_mask = tf.logical_not(tf.cast(tf.eye(dim), dtype=tf.bool))
   #     not_nan_mask = tf.logical_not(tf.math.is_nan(ap_dist))
    #    tf_mask = tf.math.logical_and(not_diag_mask, not_nan_mask)
        inter_separation = tf.reduce_min(tf.boolean_mask(tensor=ap_dist, mask=not_diag_mask))
        return intra_class_sq_diff, inter_separation

    def inter_intra_transformation(self, data, labels, class_mean, dim):
        """ Calculates intra-class spread as average distance from class means.
        Calculates inter-class separation as the distance between the two closest class means.
        Returns:
        intra-class spread and inter-class separation.
        """
        intra_class_sq_diff, _ = self.inter_intra_diff(data, labels, class_mean, dim)
        ap_dist = self.all_pair_distance(class_mean)
        dim = tf.shape(class_mean)[0]
        not_diag_mask = tf.logical_not(tf.cast(tf.eye(dim), dtype=tf.bool))
        inter_separation = tf.reduce_mean(tf.boolean_mask(tensor=ap_dist, mask=not_diag_mask))
        return intra_class_sq_diff, inter_separation

    def triplet_center_loss(self, data, labels, class_mean, alpha=0.4):
        sq_diff = self.sq_difference_from_mean(data, class_mean, self.y_dim)
        masked_neg_sq_diff = tf.multiply(sq_diff, tf.cast(1 - labels, dtype=tf.float32))
        masked_pos_sg_diff = tf.multiply(sq_diff, tf.cast(labels, dtype=tf.float32))
        inter_min = tf.reduce_min(masked_neg_sq_diff, axis=0)
        loss = tf.reduce_sum(tf.nn.relu(masked_pos_sg_diff - inter_min + alpha))
        return loss

    def feat2prob(self, feat, center, alpha=1.0):
        q = 1.0 / (1.0 + tf.reduce_sum(
            tf.math.pow(tf.expand_dims(feat,1) - center, 2), 2) / alpha)
        q = tf.math.pow(q, (alpha + 1.0) / 2.0)
        q = tf.transpose(tf.transpose(q) / tf.reduce_sum(q, 1))
       # q = q.pow((alpha + 1.0) / 2.0)
       # q = (q.t() / tf.reduce_sum(q, 1)).t()
        return q

    def target_distribution(self, q):
        weight = q ** 2 / tf.reduce_sum(q, axis=0)
        return tf.transpose(tf.transpose(weight) / tf.reduce_sum(weight, axis=1))

    @tf.function
    def KMeansCentroids(self, data, num_clusters, steps, use_mini_batch=False):
       # data_copy = tf.(data)
        kmeans = KMeans(
            num_clusters=num_clusters,
            use_mini_batch=use_mini_batch)
        kmeans.train(input_fn=lambda: data,
                     steps=steps)
        return kmeans.cluster_centers()

    def entropy(self, x):
        """
        Helper function to compute the entropy over the batch
        input: batch w/ shape [b, num_classes]
        output: entropy value [is ideally -log(num_classes)]
        """
        EPS = 1e-8
        x_ = tf.clip_by_value(x, clip_value_min=EPS, clip_value_max=10)
        b = x_ * tf.math.log(x_)
        return -1 * tf.reduce_sum(b)

        # if len(b.size()) == 2:  # Sample-wise entropy
        #     return - b.sum(dim=1).mean()
        # elif len(b.size()) == 1:  # Distribution-wise entropy
        #     return - b.sum()
        # else:
        #     raise ValueError('Input tensor is %d-Dimensional' % (len(b.size())))

    def general_intra_loss(self, data, cluster_centers, dim):
        sq_diff = self.sq_difference_from_mean(data, cluster_centers, dim)
       # sq_diff_norm = tf.nn.softmax(sq_diff, axis=0)
     #   sq_diff_norm = self.feat2prob(data, cluster_centers)
        prob = tf.math.exp(-1 * sq_diff)
        sq_diff_norm = prob / tf.reshape(tf.reduce_sum(prob, axis=1), (-1,1))
        print('sq diff nor shape ', sq_diff_norm.shape)
        sq_weighted_diff = tf.math.multiply(sq_diff, sq_diff_norm)
        total_diff = tf.reduce_sum(sq_weighted_diff, axis=0)
        total_norm = tf.reduce_sum(sq_diff_norm, axis=0)
        general_intra = tf.math.divide(total_diff, total_norm)
     #   print('general_intra shape ', general_intra.shape)
    #    entropy = self.entropy(tf.reduce_mean(sq_diff_norm, axis=0))

        return tf.reduce_max(general_intra), sq_diff_norm

    def mag_loss(self, data, unknown_data):
        all_data = tf.concat([data, unknown_data], axis=0)
        norm = tf.norm(all_data, axis=0)
        return tf.reduce_min(norm)

    def cluster_loss(self, data, cluster_centers):
      #  def input_fn():
       #     return data
        # cluster_centers = self.KMeansCentroids(data, num_clusters, num_iterations)
        # kmeans = tf.estimator.experimental.KMeans(
        #     num_clusters=num_clusters, use_mini_batch=False)
        # for _ in xrange(num_iterations):
        #     kmeans.train(input_fn = lambda : data)
        #     cluster_centers = kmeans.cluster_centers()
        q = self.feat2prob(data, cluster_centers)
        t = self.target_distribution(q)
        k = tf.keras.losses.KLDivergence()
        kl_loss = k(q, t)
        return kl_loss
      #  cluster_indices = list(kmeans.predict_cluster_index(data))

    def inter_cluster_loss(self, class_mean, cluster_centers):
        all_mean = tf.concat((class_mean, cluster_centers), axis=0)
      #  print("all mean shape ", all_mean.shape)
        ap_dist = self.all_pair_distance(all_mean)
        dim = tf.shape(all_mean)[0]
        not_diag_mask = tf.logical_not(tf.cast(tf.eye(dim), dtype=tf.bool))

        total_len = class_mean.shape[0] + cluster_centers.shape[0]
        non_cluster_dist = [[True] * total_len] * total_len
        for i in range(class_mean.shape[0], total_len):
            for j in range(class_mean.shape[0], total_len):
                non_cluster_dist[i][j] = False
        non_cluster_mask = tf.convert_to_tensor(non_cluster_dist, dtype=tf.bool)
        tf_mask = tf.math.logical_and(not_diag_mask, non_cluster_mask)
     #   not_nan_mask = tf.math.is_finite(ap_dist)
      #  tf_mask = tf.math.logical_and(not_diag_mask, not_nan_mask)
        inter_separation = tf.reduce_min(tf.boolean_mask(tensor=ap_dist, mask=not_diag_mask))
        return inter_separation

    def all_pair_distance(self, A):
        r = tf.reduce_sum(A*A, 1)
        # turn r into column vector
        r = tf.reshape(r, [-1, 1])
        D = r - 2*tf.matmul(A, A, transpose_b=True) + tf.transpose(r)
        return D

    def all_center_distance(self, A):
        center_diff_list = []

        centroid = tf.reduce_mean(A, 0)
       # data = [0,0,0,0,0,0]
       # data_np = np.asarray(data, np.float32)
       # data_tf = tf.convert_to_tensor(data_np, np.float32)
        data_tf = tf.fill(tf.shape(centroid), 0.0)
        dist_zeros = tf.squared_difference(data_tf, centroid)
      #  dist_zeros = np.sum(np.square(centroid - data_tf))

        for i in range(self.y_dim):
      #      mask_not_i = [True] * self.y_dim
       #     mask_not_i[i] = False
        #    center_not_i = tf.boolean_mask(A, np.array(mask_not_i))
         #   other_centroid = tf.reduce_mean(center_not_i, 0)
            #center_diff_list.append(tf.norm(A[i], ord='euclidean'))

           # center_diff_list.append(np.sum(np.square(A[i] - data_tf)))
            center_diff_list.append(tf.squared_difference(A[i], centroid))

        return tf.reduce_min(center_diff_list), tf.reduce_mean(dist_zeros)


    def feature_space_separation_2(self, A):
        square = tf.norm(A, axis=1)
        return tf.reduce_mean(square), tf.reduce_min(square)

    def feature_space_separation(self, A):
        absA = tf.math.abs(A)
        negA = tf.math.negative(A)
        high_signature = tf.reduce_max(absA, 1)
        low_signature = tf.reduce_min(absA, 1)
       # high_signature_1 = tf.reduce_max(A, 1)
       # high_signature_2 = tf.reduce_max(negA, 1)
        return tf.reduce_min(high_signature), tf.reduce_max(low_signature)

    def all_pair_inter_intra_diff(self, xs, ys):
        """ Calculates the intra-class and inter-class distance
        as the average distance between all pair of instances intra and inter class
        instances.
        """
        def outer(ys):
            return tf.matmul(ys, ys, transpose_b=True)

        ap_dist = self.all_pair_distance(xs)
        mask = outer(ys)

        dist = self.bucket_mean(ap_dist, mask, 2)
        inter_class_sq_diff = dist[0]
        intra_class_sq_diff = dist[1]
        return intra_class_sq_diff, inter_class_sq_diff

    def build_model(self):
        """ Builds the network graph.
        """
        pass

    def loss_fn_training_op(self, x, y, z, h, decouple, logits, x_recon, x_decouple_recon, class_means, rot, trans_means=None,
                            rot_logits=None, d_z=None, t_z=None, projection=None, x_unknown=None, y_unknown=None,
                            z_unknown=None, d_z_unknown=None, center_unknown=None, recon_unknown=None, rot_unknown=None, unknown_logits=None):
        """ Computes the loss functions and creates the update ops.

        :param x - input X
        :param y - labels y
        :param z - z layer transform of X.
        :param logits - softmax logits if ce loss is used. Can be None if only ii-loss.
        :param recon - reconstructed X. Experimental! Can be None.
        :class_means - the class means.
        """
        # Calculate intra class and inter class distance
        if self.z_dim <= 8:
            if self.dist == 'class_mean':   # For experimental pupose only
                self.intra_c_loss, self.inter_c_loss = self.inter_intra_diff(
                    z, tf.cast(y, tf.int32), class_means, self.y_dim)
            elif self.dist == 'all_pair':   # For experimental pupose only
                self.intra_c_loss, self.inter_c_loss = self.all_pair_inter_intra_diff(
                    z, tf.cast(y, tf.int32))
            elif self.dist == 'mean_separation_spread':  # ii-loss
            #    z_origin = z[0::2]
             #   y_origin = y[0::2]
                self.intra_c_loss, self.inter_c_loss = self.inter_separation_intra_spred(
                    z, tf.cast(y, tf.int32), class_means, self.y_dim)
            elif self.dist == 'min_max':   # For experimental pupose only
                self.intra_c_loss, self.inter_c_loss = self.inter_min_intra_max(
                    z, tf.cast(y, tf.int32), class_means, self.y_dim)
            elif self.dist == 'triplet_center_loss':
                self.tc_loss = tf.contrib.losses.metric_learning.triplet_semihard_loss(tf.argmax(tf.cast(y,tf.int32), axis=1), z, margin=self.margin)

            if self.enable_intra_sample_loss:
                self.intra_sample_loss = self.intra_sample_distance3(z)

            if self.enable_cluster_loss:
                self.cluster_loss, self.sq_diff_norm = self.general_intra_loss(z_unknown, center_unknown, center_unknown.shape[0])
              #  self.cluster_loss = self.cluster_loss(z_unknown, center_unknown)
                self.inter_cc_loss = self.inter_cluster_loss(class_means, center_unknown)
                # cluster_loss_1 = self.cluster_loss(z_unknown, center_unknown)
                # cluster_loss_2 = self.inter_cluster_loss(class_means, center_unknown)
                # self.cluster_loss = tf.reduce_mean(cluster_loss_1 - 0.01 * cluster_loss_2)

        else:
            if self.dist == 'class_mean':   # For experimental pupose only
                self.intra_c_loss, self.inter_c_loss = self.inter_intra_diff(
                    d_z, tf.cast(y, tf.int32), class_means, self.y_dim)
            elif self.dist == 'all_pair':   # For experimental pupose only
                self.intra_c_loss, self.inter_c_loss = self.all_pair_inter_intra_diff(
                    d_z, tf.cast(y, tf.int32))
            elif self.dist == 'mean_separation_spread':  # ii-loss
                self.intra_c_loss, self.inter_c_loss = self.inter_separation_intra_spred(
                    d_z, tf.cast(y, tf.int32), class_means, self.y_dim)
            elif self.dist == 'min_max':   # For experimental pupose only
                self.intra_c_loss, self.inter_c_loss = self.inter_min_intra_max(
                    d_z, tf.cast(y, tf.int32), class_means, self.y_dim)
            elif self.dist == 'triplet_center_loss':
                self.tc_loss = tf.contrib.losses.metric_learning.triplet_semihard_loss(tf.argmax(tf.cast(y,tf.int32), axis=1), d_z, margin=self.margin)

            if self.enable_intra_sample_loss:
                self.intra_sample_loss = self.intra_sample_distance3(d_z)
            if self.enable_cluster_loss:
                self.cluster_loss, self.sq_diff_norm = self.general_intra_loss(d_z_unknown, center_unknown, center_unknown.shape[0])
              #  self.cluster_loss = self.cluster_loss(d_z_unknown, center_unknown)
                self.inter_cc_loss = self.inter_cluster_loss(class_means, center_unknown)


        if self.enable_mmf_extension and self.enable_cluster_loss:
            self.max_feature, self.min_feature = self.feature_space_separation(tf.concat([class_means, center_unknown], 0))
        if self.enable_decouple:
            self.decouple_loss = tf.reduce_mean(tf.squared_difference(x, x_decouple_recon))
        if self.enable_recon_loss:    # For experimental pupose only
            self.recon_loss = tf.reduce_mean(tf.squared_difference(rot, x_recon))
        if self.enable_intra_loss and self.enable_inter_loss and not self.enable_mmf_extension and not \
                self.enable_tc_loss and not self.enable_self and not self.enable_cluster_loss:        # The correct ii-loss
            self.loss = tf.reduce_mean(self.intra_c_loss - self.inter_c_loss)
        elif self.enable_intra_loss and self.enable_inter_loss and self.enable_cluster_loss and not self.enable_mmf_extension and not self.enable_ce_loss:        # The correct ii-loss
            self.loss = tf.reduce_mean(self.cluster_loss +  self.intra_c_loss -  self.inter_cc_loss)
        elif self.enable_intra_loss and self.enable_inter_loss and self.enable_cluster_loss and not self.enable_mmf_extension and self.enable_ce_loss:  # The correct ii-loss
            self.ce_loss = self.entropy(tf.reduce_mean(self.sq_diff_norm, axis=0))
            self.loss = tf.reduce_mean( self.cluster_loss +  self.intra_c_loss - self.inter_cc_loss - self.ce_loss)
        elif self.enable_intra_loss and self.enable_inter_loss and self.enable_cluster_loss and self.enable_mmf_extension:        # The correct ii-loss
            self.loss = tf.reduce_mean(self.cluster_loss +  self.intra_c_loss -  self.inter_cc_loss + self.min_feature - self.max_feature)
     #   elif self.enable_intra_loss and self.enable_inter_loss and self.enable_cluster_loss:        # The correct ii-loss
      #      self.loss = tf.reduce_mean(self.cluster_loss + 0.001 * self.intra_c_loss - 0.01 * self.inter_cc_loss + self.recon_loss)
        elif self.enable_tc_loss and self.enable_mmf_extension:
            self.loss = tf.reduce_mean(self.tc_loss  - self.comb * self.max_feature  + self.comb * self.min_feature )
        elif self.enable_intra_loss and not self.enable_inter_loss:  # For experim ental pupose only
            self.loss = tf.reduce_mean(self.intra_c_loss)
        elif self.enable_tc_loss and not self.enable_mmf_extension:
           # self.tc_loss = SELF.TRIPLET_CENTER_LOSS(Z, TF.CAST(Y, TF.INT32), CLASS_MEANS)
            self.loss = tf.reduce_mean(self.tc_loss)
        elif not self.enable_intra_loss and self.enable_inter_loss:  # For experimental pupose only
            self.loss = tf.reduce_mean(-self.inter_c_loss)
      #  elif self.enable_intra_loss and self.enable_inter_loss and self.enable_self:
        #    self.loss = tf.reduce_mean(self.intra_c_loss - self.inter_c_loss +  self.self_sup)
        elif self.enable_intra_loss and self.enable_inter_loss and self.enable_mmf_extension and not self.enable_tc_loss:
            print("inter_c_loss loss: ")
            print(self.inter_c_loss)
           # self.loss = tf.reduce_mean(self.intra_c_loss - self.inter_c_loss) - self.comb * self.mmf_extension
            self.loss = tf.reduce_mean(self.intra_c_loss - self.inter_c_loss  - self.comb * self.max_feature  + self.comb * self.min_feature)
        elif self.div_loss:                                          # For experimental pupose only
            self.loss = tf.reduce_mean(self.intra_c_loss / self.inter_c_loss)
        elif self.enable_mmf_extension and not self.enable_inter_loss and not self.enable_intra_loss:
          #  self.loss = tf.reduce_mean( self.min_feature - self.max_feature)
            self.loss = tf.reduce_mean( - self.max_feature)

        else:                                                        # For experimental pupose only
            self.loss = tf.reduce_mean((self.recon_loss * 1. if self.enable_recon_loss else 0.)
                                       + (self.intra_c_loss * 1. if self.enable_intra_loss else 0.)
                                       - (self.inter_c_loss * 1. if self.enable_inter_loss else 0.)
                                      )

        # Classifier loss
        if self.enable_cluster_loss and self.enable_ce_loss:
            self.ce_loss = self.entropy(tf.reduce_mean(unknown_logits, axis=0))

        if self.enable_ce_loss:
            self.ce_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.y))

        if self.enable_trans_ce_loss:
            self.trans_ce_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=rot_logits, labels=self.decouple))

        if self.enable_self:
            self.self_sup = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=rot_logits, labels=self.decouple))
       # if self.enable_ce_loss and self.enable_ce_loss:
          #  self.ce_self_loss = tf.reduce_mean(self.ce_loss + 0.2 * self.self_sup)

        self.tvars = tf.trainable_variables()
        e_vars = [var for var in self.tvars if 'enc_' in var.name]
     #   e_h_vars = [var for var in self.tvars if 'enc_' in var.name or 'ss_h' in var.name]
        self.pre_var = [var for var in self.tvars if 'enc_' in var.name]
        classifier_vars = [var for var in self.tvars if 'enc_' in var.name or 'classifier_' in var.name]
        rot_vars = [var for var in self.tvars if 'enc_' in var.name or 'rot_' in var.name]
        recon_vars = [var for var in self.tvars if 'enc' in var.name or 'dec_' in var.name]
        classifier_rot_vars = [var for var in self.tvars if 'enc_' in var.name or 'rot_' in var.name]
        decouple_vars = [var for var in self.tvars if 'enc_' in var.name or 'decouple_' in var.name]
        projection_vars = [var for var in self.tvars if 'enc_' in var.name or 'projection' in var.name]
        # Training Ops

        if self.enable_decouple:
            self.decouple_train_op = self.opt.minimize(self.decouple_loss, var_list=decouple_vars)
        if self.enable_recon_loss:
            self.recon_train_op = self.recon_opt.minimize(self.recon_loss, var_list=recon_vars)
       #     self.iir_op = self.opt.minimize(self.iir_loss, var_list=e_vars)
        if self.enable_intra_sample_loss:
            self.intra_sample_train_op = self.opt.minimize(self.intra_sample_loss, var_list=e_vars)

        if self.enable_inter_loss or self.enable_intra_loss or self.div_loss or self.enable_mmf_extension or self.enable_tc_loss:
            self.train_op = self.opt.minimize(self.loss, var_list=classifier_vars)

        if self.enable_ce_loss:
            self.ce_train_op = self.c_opt.minimize(self.ce_loss, var_list=classifier_vars)

        if self.enable_trans_ce_loss:
            self.trans_ce_op = self.rot_opt.minimize(self.trans_ce_loss, var_list=classifier_rot_vars)

        # if self.enable_self and self.enable_ce_loss:
        #     self.self_train_op = self.rot_opt.minimize(self.ce_loss + self.comb * self.
        #
        #     , var_list=classifier_rot_vars)
        #
        if self.enable_self:
            self.self_train_op =  self.rot_opt.minimize(self.self_sup, var_list=classifier_vars)

      #  if self.enable_cluster_loss:
       #     self.cluster_train_op = self.opt.minimize(self.cluster_loss, var_list=e_vars)

    def fit(self, X, y, x_unknown=None, y_unknown=None, X_val=None, y_val=None, plot=False,
            figsize=[6,4], loc = 'lower right', rot=None, decouple=None, rot_unknown=None, decouple_unknown=None):
        """ Fit model.
        """

        assert y.shape[1] == self.y_dim

        start = time.time()
        print("start time: " + str(start))
        self.is_training = True
        y_count_skip = 0

        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
            if self.load_model_directory is not None:
                self.saver = tf.train.Saver(self.pre_var)
                self.saver.restore(self.sess, self.load_model_directory)
                print("Model restored.")

            i = 0
            ii_loss_arr = []
            mmf_arr = []
            iter_arr = []
         #   with tf.variable_scope('enc_z', reuse=True):
         #       weights = tf.get_variable('weights')
          #      weights_v = self.sess.run(weights)
          ##      bias = tf.get_variable('bias')
           #     bias_v = self.sess.run(bias)
           #     print(weights_v)
          #      print(bias_v)
            cluster_y, cc_means = None, None

            while i < self.iterations:

                if self.enable_cluster_loss and (i in range(0, 1000, 10)+range(1000, self.iterations, 100)):
                    cluster_y, cc_means = self.assign_clusters(x_unknown)

                if not (decouple is None) and not (rot is None):
                    xs, ys, rots, decouples = self._next_batch(X, y, rot, decouple)
                 #   print("decouple type ", type(xs))
                    if type(self.rot_dim) is not int:
                        rots = self.x_reformat(rots)
                elif decouple is None and not (rot is None):
                    xs, ys, rots = self._next_batch(X, y, rot)
                    if type(self.rot_dim) is not int:
                        rots = self.x_reformat(rots)
                else:
                    xs, ys = self._next_batch(X, y)
                if self.enable_cluster_loss:
                    if self.enable_recon_loss:
                        xs_unknown, ys_unknown, rots_unknown = self._next_batch(x_unknown, y_unknown, rot_unknown)
                        rots_unknown = self.x_reformat(rots_unknown)
                    else:
                        xs_unknown, ys_unknown = self._next_batch(x_unknown, y_unknown)
                    xs_unknown = self.x_reformat(xs_unknown)
                xs = self.x_reformat(xs)


              #  print('xs shape ', xs.shape)

              #  print('here 3')
         #       print("rots shape: ", len(rots[0]))

                self_sup, intra_c_loss, inter_c_loss, recon_loss, loss, ce_loss, max_feature, min_feature, tc_loss, acc, \
                val_acc, decouple_loss, trans_ce_loss, intra_sample_loss, cluster_loss= \
                    None, None, None, None, None, None, None, None, None, None, None, None, None, None, None

                if  len(np.unique(np.argmax(ys, axis=1))) != self.y_dim:
                    y_count_skip += 1
                    continue

                # if (not self.enable_cluster_loss and len(np.unique(np.argmax(ys, axis=1))) != self.y_dim) or \
                #         (self.enable_cluster_loss and len(np.unique(np.argmax(ys, axis=1))) + len(np.unique(np.argmax(ys_unknown, axis=1))) != self.y_dim) :
                #     print(len(np.unique(np.argmax(ys, axis=1))))
                #     y_count_skip += 1
                #     continue


                if self.enable_self and not (self.enable_ce_loss or self.enable_inter_loss or self.enable_intra_loss or self.div_loss or self.enable_tc_loss or self.enable_mmf_extension):
                    _, self_sup = self.sess.run(
                        [self.self_train_op, self.self_sup],
                        feed_dict={self.x: xs, self.decouple: decouples})

                if self.enable_inter_loss or self.enable_intra_loss or self.div_loss or self.enable_tc_loss or self.enable_mmf_extension:
                    if self.enable_tc_loss and not self.enable_mmf_extension:
                        _, tc_loss, loss = self.sess.run(
                            [self.train_op, self.tc_loss, self.loss],
                            feed_dict={self.x:xs, self.y:ys})

                    elif self.enable_tc_loss and self.enable_mmf_extension:
                        _, tc_loss, max_feature, min_feature, loss = self.sess.run(
                            [self.train_op, self.tc_loss, self.max_feature, self.min_feature, self.loss],
                            feed_dict={self.x:xs, self.y:ys})

                    elif self.enable_intra_loss and self.enable_inter_loss and self.enable_mmf_extension and not self.enable_tc_loss and not self.enable_cluster_loss:
                        _, intra_c_loss, inter_c_loss, max_feature, min_feature, loss = self.sess.run(
                            [self.train_op, self.intra_c_loss, self.inter_c_loss, self.max_feature, self.min_feature, self.loss],
                            feed_dict={self.x:xs, self.y:ys})
                    elif not self.enable_intra_loss and not self.enable_inter_loss and self.enable_mmf_extension and not self.enable_tc_loss:
                        _, max_feature,  min_feature, loss = self.sess.run(
                            [self.train_op, self.max_feature, self.min_feature, self.loss],
                            feed_dict={self.x:xs, self.y:ys})

                    elif self.enable_intra_loss and self.enable_inter_loss and not self.enable_self and not self.enable_cluster_loss:
                        _, intra_c_loss, inter_c_loss, loss = self.sess.run(
                            [self.train_op, self.intra_c_loss, self.inter_c_loss, self.loss],
                            feed_dict={self.x:xs, self.y: ys})

                    elif self.enable_intra_loss and self.enable_inter_loss and self.enable_cluster_loss and not self.enable_mmf_extension and not self.enable_ce_loss:
                        _, cluster_loss, intra_c_loss, inter_c_loss, loss = self.sess.run(
                            [self.train_op, self.cluster_loss, self.intra_c_loss, self.inter_cc_loss, self.loss],
                            feed_dict={self.x:xs, self.y: ys, self.x_unknown:xs_unknown, self.center_unknown: cc_means})

                    elif self.enable_intra_loss and self.enable_inter_loss and self.enable_cluster_loss and not self.enable_mmf_extension:
                        _, cluster_loss, intra_c_loss, inter_c_loss, ce_loss, loss = self.sess.run(
                            [self.train_op, self.cluster_loss, self.intra_c_loss, self.inter_cc_loss, self.ce_loss, self.loss],
                            feed_dict={self.x:xs, self.y: ys, self.x_unknown:xs_unknown, self.center_unknown: cc_means})

                    elif self.enable_intra_loss and self.enable_inter_loss and self.enable_cluster_loss and self.enable_mmf_extension:
                        _, cluster_loss, intra_c_loss, inter_c_loss,  max_feature,  min_feature, loss = self.sess.run(
                            [self.train_op, self.cluster_loss, self.intra_c_loss, self.inter_cc_loss, self.max_feature, self.min_feature,  self.loss],
                            feed_dict={self.x:xs, self.y: ys, self.x_unknown:xs_unknown, self.center_unknown: cc_means})

                    # elif self.enable_intra_loss and self.enable_inter_loss and self.enable_cluster_loss and self.enable_recon_loss:
                    #     _, cluster_loss, intra_c_loss, inter_c_loss, recon_loss, loss = self.sess.run(
                    #         [self.train_op, self.cluster_loss, self.intra_c_loss, self.inter_cc_loss, self.recon_loss, self.loss],
                    #         feed_dict={self.x:xs, self.y: ys, self.x_unknown:xs_unknown, self.center_unknown: c_unknown, self.rot_unknown: rots_unknown})
                    #

                if not self.enable_recon_loss and self.enable_trans_ce_loss:
                    _, self_sup = self.sess.run(
                        [self.self_train_op, self.self_sup],
                        feed_dict={self.x:xs, self.rot: rots})

                if self.enable_recon_loss and not self.enable_trans_ce_loss:
                    _,  recon_loss = self.sess.run(
                        [self.recon_train_op, self.recon_loss],
                        feed_dict={ self.x_unknown: xs_unknown, self.rot_unknown: rots_unknown})

              #         elif self.enable_intra_loss and self.enable_inter_loss:
                     #   _, intra_c_loss, inter_c_loss, loss = self.sess.run(
                    #        [self.train_op, self.intra_c_loss, self.inter_c_loss, self.loss],
                      #      feed_dict={self.x:xs, self.y:ys})
                #
                # elif self.enable_self and (self.enable_inter_loss or self.enable_intra_loss or self.div_loss or self.enable_tc_loss or self.enable_mmf_extension):
                #     if self.enable_tc_loss and not self.enable_mmf_extension:
                #         _, self_sup, tc_loss, loss = self.sess.run(
                #             [self.self_train_op, self.self_sup, self.tc_loss, self.loss],
                #             feed_dict={self.x:xs, self.y:ys, self.rot:rots})
                #
                #     elif self.enable_tc_loss and self.enable_mmf_extension:
                #         _, self_sup, tc_loss, max_feature, min_feature, loss = self.sess.run(
                #             [self.self_train_op, self.self_sup, self.tc_loss, self.max_feature, self.min_feature, self.loss],
                #             feed_dict={self.x:xs, self.y:ys, self.rot:rots})
                #
                #     elif self.enable_intra_loss and self.enable_inter_loss and self.enable_mmf_extension and not self.enable_tc_loss:
                #         _, self_sup, intra_c_loss, inter_c_loss, max_feature, min_feature, loss = self.sess.run(
                #             [self.self_train_op, self.self_sup, self.intra_c_loss, self.inter_c_loss, self.max_feature, self.min_feature, self.loss],
                #             feed_dict={self.x:xs, self.y:ys, self.rot:rots})
                #     elif not self.enable_intra_loss and not self.enable_inter_loss and self.enable_mmf_extension and not self.enable_tc_loss:
                #         _, self_sup, max_feature,  min_feature, loss = self.sess.run(
                #             [self.self_train_op, self.self_sup, self.max_feature, self.min_feature, self.loss],
                #             feed_dict={self.x:xs, self.y:ys, self.rot:rots})
                #     else:
                #         _, self_sup, intra_c_loss, inter_c_loss, loss = self.sess.run(
                #             [self.self_train_op, self.self_sup, self.intra_c_loss, self.inter_c_loss, self.loss],
                #             feed_dict={self.x:xs, self.y:ys, self.rot:rots})


                    # _, iir_loss = self.sess.run(
                    #     [self.iir_op, self.iir_loss], feed_dict={self.x:xs, self.decouple:decouples}
                    # )

                if self.enable_recon_loss and not self.enable_self:
                    _, recon_loss = self.sess.run(
                        [self.recon_train_op, self.recon_loss],
                        feed_dict={self.x:xs, self.rot: rots})

          #      if self.enable_self:
          #           _, self_sup = self.sess.run(
            #             [self.self_train_op, self.self_sup],
             #            feed_dict={self.x: xs, self.rot: rots})
                if self.enable_intra_sample_loss:
                    _, intra_sample_loss = self.sess.run(
                        [self.intra_sample_train_op, self.intra_sample_loss],
                        feed_dict={self.x:xs}
                    )

                if self.enable_decouple:
                    _, decouple_loss = self.sess.run(
                        [self.decouple_train_op, self.decouple_loss],
                        feed_dict={self.x:xs}
                    )

                if self.enable_trans_ce_loss:
                    _, trans_ce_loss = self.sess.run(
                        [self.trans_ce_op, self.trans_ce_loss],
                        feed_dict={self.x:xs, self.decouple: decouples}
                    )

                if self.enable_ce_loss and not self.enable_self:
                    _, ce_loss, acc = self.sess.run(
                        [self.ce_train_op, self.ce_loss, self.acc],
                        feed_dict={self.x:xs, self.y:ys})
                    if X_val is not None and y_val is not None:
                        # val_acc = self.sess.run(
                        #     self.acc,
                        #     feed_dict={self.x:self.x_reformat(X_val), self.y:y_val})
                        val_acc = (self.predict(X_val) == np.argmax(y_val, axis=1)).astype(np.float).mean()

                # if self.enable_cluster_loss:
                #   #  print('x unknown type', xs_unknown.dtype)
                #     _, cluster_loss = self.sess.run(
                #         [self.cluster_train_op, self.cluster_loss],
                #         feed_dict={self.x:xs, self.x_unknown:xs_unknown, self.y:ys, self.center_unknown: c_unknown}
                #     )


                # if self.enable_self and self.enable_ce_loss:
                #     _, self_sup, ce_loss, acc = self.sess.run(
                #         [self.self_train_op, self.self_sup, self.ce_loss, self.acc],
                #         feed_dict={self.x:xs, self.y:ys, self.rot:rots})
                #     if X_val is not None and y_val is not None:
                #         val_acc = (self.predict(X_val) == np.argmax(y_val, axis=1)).astype(np.float).mean()

                if i % self.display_step == 0 and self.display_step > 0:
                    if (self.enable_inter_loss and self.enable_intra_loss) or self.div_loss or self.enable_tc_loss or self.enable_mmf_extension:
                        if not (decouple is None):
                            self.update_class_stats(X, y, decouple)
                        elif not (x_unknown is None):
                        #    cluster_y, cc_means = self.assign_clusters(x_unknown)
                            self.update_class_stats(X, y, x_unknown=x_unknown, y_unknown=cluster_y, unknown_means=cc_means)
                        else:
                            self.update_class_stats(X, y)
                        acc = (self.predict(xs, reformat=False) == np.argmax(ys, axis=1)).mean()
                        if X_val is not None and y_val is not None:
                            val_acc = (self.predict(X_val) == np.argmax(y_val, axis=1)).astype(np.float).mean()

                    self._iter_stats(i, start,  intra_c_loss, inter_c_loss, recon_loss, decouple_loss, trans_ce_loss, tc_loss, intra_sample_loss, ce_loss,
                                     acc, val_acc, self_sup, cluster_loss)


                if i % self.save_step == 0 and i != 0 and self.model_directory is not None:
                    self.save_model('model-'+str(i)+'.cptk')
                    print ("Saved Model")

          #      if plot and i % self.display_step == 0:
           #         iter_arr.append(i)
             #       ii_loss_arr.append(intra_c_loss - inter_c_loss)
             #       mmf_arr.append(mmf_extension)

                i += 1


        if self.display_step > 0:
            if (self.enable_inter_loss and self.enable_intra_loss) or self.div_loss:
                if not (decouple is None):
                    self.update_class_stats(X, y, decouple)
                elif not (x_unknown is None):
               #     cluster_y, cc_means = self.assign_clusters(x_unknown)
                    self.update_class_stats(X, y, x_unknown=x_unknown, y_unknown=cluster_y, unknown_means=cc_means)
                else:
                    self.update_class_stats(X, y)
                acc = (self.predict(xs, reformat=False) == np.argmax(ys, axis=1)).mean()
                if X_val is not None and y_val is not None:
                    val_acc = (self.predict(X_val) == np.argmax(y_val, axis=1)).mean()

            self._iter_stats(i, start,  intra_c_loss, inter_c_loss, recon_loss, decouple_loss, trans_ce_loss, tc_loss, intra_sample_loss, ce_loss, acc, val_acc, self_sup, cluster_loss)
        if self.model_directory is not None:
            self.save_model('model-'+str(i)+'.cptk')
            print ("Saved Model")

        # Save class means and cov

        if not (decouple is None):
            self.update_class_stats(X, y, dec=decouple)
        elif not (x_unknown is None):
          #  cluster_y, cc_means = self.assign_clusters(x_unknown)
            self.update_class_stats(X, y, x_unknown=x_unknown, y_unknown=cluster_y, unknown_means=cc_means)
        else:
            self.update_class_stats(X, y)


        # Compute and store the selected thresholds for each calls
     #   self.class_thresholds_std_global(X, y)
        if self.enable_cluster_loss:
       #     self.class_thresholds_std_global()
            self.class_thresholds(X, y, x_unknown, cluster_y, cc_means)
        else:
         #   self.class_thresholds_std_global()
            self.class_thresholds(X, y)
#         print ('n_skipped_batches = ', count_skip)

        self.is_training = False

        if plot:
            plt.figure(figsize=figsize)
            plt.plot(iter_arr, ii_loss_arr, label='ii loss')
            plt.plot(iter_arr, mmf_arr, label='mmf extension')
            plt.grid(True)
            plt.xlabel('iterations')
            plt.ylabel('loss')
            plt.xlim([0., self.iterations])
            plt.legend(loc=loc)
            plt.show()

    def distance_to_centroids(self, A, B):
     #   assert A.shape.as_list() == B.shape.as_list()

        row_norms_A = tf.reduce_sum(tf.square(A), axis=1)
        row_norms_A = tf.reshape(row_norms_A, [-1, 1])  # Column vector.

        row_norms_B = tf.reduce_sum(tf.square(B), axis=1)
        row_norms_B = tf.reshape(row_norms_B, [1, -1])  # Row vector.
        return row_norms_A - 2 * tf.matmul(A, tf.transpose(B)) + row_norms_B

    def assign_clusters(self, X):
        z = self.latent(X)
        if self.z_dim > 8:
            z = z[:,:-3]

        kmeans = KMeans(n_clusters=2, n_init=20)
        y_pred = kmeans.fit_predict(z)

 #       dist = euclidean_distances(z, self.cc_means)
       # print('pair shape ', dist.shape)
#        assigments = np.argmin(dist, axis=1)
        one_hot = np.zeros((y_pred.size, y_pred.max()+1))
        one_hot[np.arange(y_pred.size),y_pred] = 1
       # print('one hot ', one_hot.shape)
        return one_hot, kmeans.cluster_centers_

    def update_class_stats(self, X, y, dec=None, x_unknown=None, y_unknown=None, unknown_means=None):
        """ Recalculates class means.
        """
        z = self.latent(X)
        self.is_training = False

        # No need to feed z_test here. because self.latent() already used z_test
        self.c_means = self.sess.run(self.class_means,
                                     feed_dict={self.z:z, self.y:y})
        self.cc_means = unknown_means
        self.t_means = None
        if not (dec is None):
            self.t_means = self.sess.run(self.trans_means,
                                     feed_dict={self.z:z, self.decouple:dec})

        if self.enable_cluster_loss:
            z_unknown = self.latent(x_unknown)

            z_full = np.concatenate((z, z_unknown), axis=0)
            y = np.concatenate((y, np.zeros((y.shape[0], self.n_clusters))), axis=1)
            #      print('y unknown shape ', y_unknown.shape)
            y_unknown = np.concatenate((np.zeros((y_unknown.shape[0], self.y_dim)), y_unknown), axis=1)
            #     print('y unknown shape ', y_unknown.shape)
            y_full = np.concatenate((y, y_unknown), axis=0)

        if self.z_dim > 8:
            z = z[:,:-3]
            if self.enable_cluster_loss:
                z_full = z_full[:, :-3]
        if self.decision_dist_fn == 'mahalanobis':
            if self.enable_cluster_loss:
                self.c_cov, self.c_cov_inv = self.class_covarience(z_full, y_full)
            else:
                self.c_cov, self.c_cov_inv = self.class_covarience(z, y)

        self.is_training = True

        # calculate class std
        if self.enable_cluster_loss:

            dist = np.zeros((z_full.shape[0], self.y_dim+self.n_clusters))
            means = np.concatenate((self.c_means, unknown_means), axis=0)
            for j in range(self.y_dim+self.n_clusters):
                if self.decision_dist_fn == 'mahalanobis':
                    dist[:, j] = scipy.spatial.distance.cdist(
                        z_full, means[j][None, :], 'mahalanobis',
                        VI=self.c_cov_inv[j]).reshape((z_full.shape[0]))
                else:
                    dist[:, j] = np.sum(np.square(z_full - means[j]), axis=1)
            dist_y = np.multiply(dist, y_full)
            dist_y[dist_y==0] = 'nan'

        else:
            dist = np.zeros((z.shape[0], self.y_dim))
            for j in range(self.y_dim):
                if self.decision_dist_fn == 'euclidean': # squared euclidean
                    dist[:, j] = np.sum(np.square(z - self.c_means[j]), axis=1)
                elif self.decision_dist_fn == 'mahalanobis':
                    dist[:, j] = scipy.spatial.distance.cdist(
                        z, self.c_means[j][None, :],
                        'mahalanobis', VI=self.c_cov_inv[j]).reshape((z.shape[0]))
                else:
                    ValueError('Error: Unsupported decision_dist_fn "{0}"'.format(self.decision_dist_fn))

            dist_y = np.multiply(dist, y)
            dist_y[dist_y==0] = 'nan'

        self.dist_mean = np.nanmean(dist_y, axis=0)
        self.dist_std = np.nanstd(dist_y, axis=0)
        #

    def class_covarience(self, Z, y):
        dim = 8
        per_class_cov = np.zeros((y.shape[1], dim, dim))
        per_class_cov_inv = np.zeros_like(per_class_cov)
        for c in range(y.shape[1]):
            per_class_cov[c, :, :] = np.cov((Z[y[:, c].astype(bool)]).T)
            per_class_cov_inv[c, :, :] = np.linalg.pinv(per_class_cov[c, :, :])

        return per_class_cov, per_class_cov_inv

    def _iter_stats(self, i, start_time,  intra_c_loss, inter_c_loss, recon_loss, max_feature, min_feature, tc_loss, loss, ce_loss, acc, val_acc, self_sup, cluster_loss):
        if i == 0:

            print '{0:5}|{1:7}|{2:7}|{3:7}|{4:7}|{5:7}|{6:7}|{7:7}|{8:7}|{9:7}|{10:7}|{11:7}|{12:7}|{13:7}'.format(
                'i', 'Self', 'Intra', 'Inter', 'Recon', 'Decouple', 'TransCE','TCL', 'IntraSample', 'ClusterL', 'CrossE', 'Acc', 'V_Acc', 'TIME(s)')

        print '{0:5}|{1:7.4}|{2:7.4}|{3:7.4}|{4:7.4}|{5:7.4}|{6:7.4}|{7:7.4}|{8:7.4}|{9:7.4}|{10:7.4}|{11:7.4}|{12:7.4}|{13:7}'.format(
            i,  self_sup, intra_c_loss, inter_c_loss, recon_loss, max_feature, min_feature, tc_loss, loss, cluster_loss, ce_loss, acc, val_acc,
            int(time.time()-start_time))

    def latent(self, X, reformat=True):
        """ Computes the z-layer output.
        """
        self.is_training = False
        z = np.zeros((X.shape[0], self.z_dim))
      #  if self.z_dim > 6:
       #     z = z[:,:-4]
        batch = self.batch_size
       # print('reformat ', reformat)
        with self.graph.as_default():
            for i in range(0, X.shape[0], batch):
                start = i
                end =  min(i+batch, X.shape[0])
                z[start: end] = self.sess.run(
                    self.z_test, feed_dict={self.x:self.x_reformat(X[start: end]) if reformat else X[start: end]})

        self.is_training = True
        return z

    def reconstruct(self, X):
        self.is_training = False
        with self.graph.as_default():
            x_recon =  self.sess.run(self.x_recon_test,
                                     feed_dict={self.x: self.x_reformat(X)})
        self.is_training = True
        return x_recon

    def distance_from_all_classes(self, X, reformat=True):
        """ Computes the distance of each instance from all class means.
        """
        means = self.c_means
        if self.enable_cluster_loss:
            means = np.concatenate((self.c_means, self.cc_means), axis=0)
        z = self.latent(X, reformat=reformat)
        if self.z_dim > 8:
            z = z[:,:-3]
        dist = np.zeros((z.shape[0], means.shape[0]))
  #       print('dist shape ', dist.shape)
        for j in range(means.shape[0]):
            if self.decision_dist_fn == 'euclidean': # squared euclidean
                dist[:, j] = np.sum(np.square(z - means[j]), axis=1)
            elif self.decision_dist_fn == 'mahalanobis':
                dist[:, j] = scipy.spatial.distance.cdist(
                    z, means[j][None, :],
                    'mahalanobis', VI=self.c_cov_inv[j]).reshape((z.shape[0]))
            else:
                ValueError('Error: Unsupported decision_dist_fn "{0}"'.format(self.decision_dist_fn))

     #   print('dist ', dist)
        return dist

    def decision_function(self, X):
        """ Computes the outlier score. The larger the score the more likely it is an outlier.(-1, 1)
        """
        dist = self.distance_from_all_classes(X)
    #    norm_dist = np.divide(np.subtract(dist,self.dist_mean), self.dist_std)
        return np.amin(dist, axis=1)

    def decision_function_std(self, X):
        dist = self.distance_from_all_classes(X)
        print('dist mean shape ', self.dist_mean.shape)
        norm_dist =  np.abs(np.divide(np.subtract(dist,self.dist_mean), self.dist_std))
        return np.amin(norm_dist, axis=1)

    def predict_prob(self, X, reformat=True):
        """ Predicts class probabilities for X over known classes.
        """
        self.is_training = False
        dist = self.distance_from_all_classes(X,  reformat=reformat)
        norm_dist = np.abs(np.divide(np.subtract(dist,self.dist_mean), self.dist_std))
        prob = np.exp(-norm_dist)
        prob = prob / prob.sum(axis=1)[:,None]
        # if self.enable_cluster_loss:
        #     known_dist = dist[:, :-1*self.n_clusters]
        #     unknown_dist = dist[:, -self.n_clusters:]
        #     known_prob = np.exp(-known_dist)
        #     known_prob = known_prob / known_prob.sum(axis=1)[:, None]
        #     unknown_prob = np.exp(-unknown_dist)
        #     unknown_prob = unknown_prob / unknown_prob.sum(axis=1)[:, None]
        #     prob = np.concatenate((known_prob, unknown_prob), axis=1)
        print('std ', self.dist_std)
        self.is_training = True
        return prob

    def predict(self, X, reformat=True):
        """ Performs closed set classification (i.e. prediction over known classes).
        """
        prob = self.predict_prob(X, reformat=reformat)
        return np.argmax(prob, axis=1)

    def predict_open(self, X, threshold=None):
        """ Performs open set recognition/classification.
        """
        pred = self.predict(X)
        pred_prob = self.predict_prob(X)
        unknown_class_label = self.y_dim + self.n_clusters
        score = self.decision_function(X)
        norm_score = self.decision_function_std(X)
     #   pred_prob = np.max(self.predict_prob(X, means), axis=1)
     #\   print('threshold 1 ', self.dist_mean)
     #   print('threshold 2 ', self.dist_std)
        for i in range(X.shape[0]):
            if not threshold:
                if 1 - np.max(pred_prob[i]) > self.threshold[pred[i]]:
                    pred[i] = unknown_class_label
            else:
                if 1 - np.max(pred_prob[i]) > threshold[pred[i]]:
                    pred[i] = unknown_class_label

        # for i in range(X.shape[0]):
        #     if not threshold:
        #         if norm_score[i] > self.threshold[pred[i]]:
        #             pred[i] = unknown_class_label
        #     else:
        #         if norm_score[i] >= threshold[pred[i]]:
        #             pred[i] = unknown_class_label
        return pred, score

    def opt_threshold(self,X, y):
        pred_prob = self.predict_prob(X)
        predictions = np.argmax(pred_prob, axis=1)
        binary_truth = np.argmax(y, axis=1)
        n_labels = len(y[0])
        result = []
        for i in range(n_labels):
            mask = (predictions == binary_truth) & (binary_truth == i)
        #    mask = binary_truth == i
            #  print(mask)
            threshold = np.min(pred_prob[mask, i])
            result.append(threshold)
        return result

    def class_threshold_dist(self, X, y):
        z = self.latent(X)
        if self.z_dim > 8:
            z = z[:,:-3]
        distance_to_c = np.zeros((z.shape[0], 1))
        truth = np.argmax(y, axis=1)
        average_intra_c = np.zeros((self.y_dim, 1))
        self.threshold = np.ones(self.y_dim)
        for i in range(z.shape[0]):
            distance_to_c[i,:] = np.linalg.norm(z[i] - self.c_means[truth[i]])
        for i in range(self.y_dim):
            mask = truth[:] == i
            average_intra_c[i] = np.max(distance_to_c[mask, :])
        c_dist = euclidean_distances(self.c_means, self.c_means)
        for i in range(self.y_dim):
            closest_c_to_i = np.argsort(c_dist[i])[1]
            dist = sorted(c_dist[i])[1]
            self.threshold[i] = (dist - average_intra_c[i] - average_intra_c[closest_c_to_i]) * average_intra_c[i] / (average_intra_c[i]+average_intra_c[closest_c_to_i]) + average_intra_c[i]
        max_t = max(self.threshold)
        self.threshold = [max_t] * self.y_dim
        return self.threshold

    def class_thresholds_std(self, X, y, std=3):
        score = self.decision_function(X)
        c_count = y.sum(axis=0)
        self.threshold = np.zeros_like(c_count)
        for c in range(y.shape[1]):
            c_scores = score[y[:, c].astype(bool)]
            score_mean = statistics.mean(c_scores)
            score_std = statistics.stdev(c_scores)
            self.threshold[c] = score_mean + std*score_std
        return self.threshold

    def class_thresholds_std_global(self, std=6):
        if self.enable_cluster_loss:
            self.threshold = np.ones(self.y_dim + self.n_clusters) * std
        else:
            self.threshold = np.ones(self.y_dim) * std
        return self.threshold

    def class_thresholds(self, X, y, x_unknown=None, y_unknown=None, cluster_center=None):
        """ Computes class thresholds. Shouldn't be called from outside.
        """
        if self.enable_cluster_loss:
            y = np.concatenate((y, np.zeros((y.shape[0], self.n_clusters))), axis=1)
            y_unknown = np.concatenate((np.zeros((y_unknown.shape[0], self.y_dim)), y_unknown), axis=1)
            x_full = np.concatenate((X, x_unknown), axis=0)
            y_full = np.concatenate((y, y_unknown), axis=0)
            means = np.concatenate((self.c_means, cluster_center), axis=0)
        else:
            x_full, y_full, means = X, y, self.c_means
        print('sq_diff_norm shape',  self.sq_diff_norm.shape)
        score = self.decision_function(x_full)
        predict_prob = self.predict_prob(x_full)
        prob = np.amax(predict_prob, axis=1)
       # print('score ', score)
        if self.threshold_type == 'global':
            self.threshold = np.ones(self.y_dim+self.n_clusters)
            cutoff_idx = max(1, int(prob.shape[0] * self.contamination))
            self.threshold *= sorted(1-prob)[-cutoff_idx]
        elif self.threshold_type == 'perclass':
            c_count = y_full.sum(axis=0)
            self.threshold = np.zeros_like(c_count)
            for c in range(y_full.shape[1]):
                sorted_c_scores = sorted(prob[y_full[:, c].astype(bool)])
                cutoff_idx = max(1, int(c_count[c] * self.contamination))
                self.threshold[c] = sorted_c_scores[-cutoff_idx]
        return self.threshold

class OpenNetFlat(OpenNetBase):
    """ OpenNet with only fully connected layers.
    """
    def __init__(self, x_dim, y_dim,
                 z_dim=6, h_dims=[128],
                 activation_fn=tf.nn.relu,
                 x_scale=UnitPosNegScale.scale,
                 x_inverse_scale=UnitPosNegScale.inverse_scale,
                 x_reshape=None,
                 opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9),
                 recon_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9),
                 c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9),
                 rot_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9),
                 dist='mean_separation_spread',
                 decision_dist_fn = 'euclidean',
                 threshold_type='perclass',
                 dropout = True, keep_prob=0.7,
                 batch_size=128, iterations=5000,
                 display_step=500, save_step=500,
                 model_directory=None,  # Directory to save trained model to.
                 density_estimation_factory=None, # Depricated
                 load_model_directory=None,
                 ce_loss=False, recon_loss=False, inter_loss=True, intra_loss=True,
                 div_loss=False,
                 combined_loss=False,
                 tc_loss=False,
                 mmf_extension=False,
                 contamination=0.02,
                 mmf_comb=0.0,
                 margin=8.0,  self_sup = False, n_rots=0
                 ):
        """
        Args:
        :param x_dim - dimension of the input
        :param y_dim - number of known classes.
        :param z_dim - the number of latent variables.
        :param h_dims - an int or a list; number of units in the fully conected hidden layers of the
                        encoder network. The decoder network (if used) will simply be the reverse.
        :param x_scale - an input scaling function. Default scale to range of [-1, 1].
                         If none, the input will not be scaled.
        :param x_inverse_scale - reverse scaling fn. by rescaling from [-1, 1] to original input scale.
                                 If None, the the output of decoder(if there is a decoder) will rescaled.
        :param x_reshape - a function to reshape the input before feeding to the networks input layer.
                            If None, the input will not be reshaped.
        :param opt - the Optimizer used when updating based on ii-loss.
                     Used when inter_loss and intra_loss are enabled. Default is AdamOptimizer.
        :param recon_opt - the Optimizer used when updating based on reconstruction-loss (Not used ii, ii+ce or ce).
                           Used when recon_loss is enabled. Default is AdamOptimizer.
        :param c_opt - the Optimizer used when updating based on cross entropy loss.
                       Used for ce and ii+ce modes (i.e. ce_loss is enabled). Default is AdamOptimizer.
        :param batch_size - training batch size.
        :param iterations - number of training iterations.
        :param display_step - training info displaying interval.
        :param save_step - model saving interval.
        :param model_directory - directory to save model in.
        :param dist - ii-loss calculation mode. Only 'mean_separation_spread' should be used.
        :param decision_dist_fn - outlier score distance functions
        :param threshold_type - outlier threshold mode. 'global' appears to give better results.
        :param ce_loss - Consider cross entropy loss. When enabled with intra_loss and inter_loss gives (ii+ce) mode.
        :param recon_loss - Experimental! Avoid enabling this.
        :param inter_loss - Consider inter-class separation. Should be enabled together with intra_loss for (ii-loss).
        :param intra_loss - Consider intra-class spread. Should be enabled together with inter_loss for (ii-loss).
        :param div_loss and combined_loss - Experimental. Avoid enabling them.
        :param contamination - contamination ratio used for outlier threshold estimation.
        """

        # Network Setting
        if isinstance(h_dims, list) or isinstance(h_dims, tuple):
            self.h_dims = h_dims
        else:
            self.h_dims = [h_dims]

        self.activation_fn = activation_fn

        assert decision_dist_fn in ['euclidean', 'mahalanobis']

        super(OpenNetFlat, self).__init__(
            x_dim, y_dim, z_dim=z_dim,
            x_scale=x_scale, x_inverse_scale=x_inverse_scale, x_reshape=x_reshape,
            opt=opt, recon_opt=recon_opt, c_opt=c_opt, rot_opt=rot_opt, threshold_type=threshold_type,
            dist=dist, decision_dist_fn=decision_dist_fn, dropout=dropout, keep_prob=keep_prob,
            batch_size=batch_size, iterations=iterations,
            display_step=display_step, save_step=save_step,
            model_directory=model_directory, load_model_directory=load_model_directory,
            ce_loss=ce_loss, recon_loss=recon_loss, inter_loss=inter_loss, intra_loss=intra_loss,
            div_loss=div_loss, mmf_extension=mmf_extension, tc_loss=tc_loss, contamination=contamination,
            mmf_comb=mmf_comb, margin=margin,  self_sup=self_sup, n_rots=n_rots)

        self.model_params += ['h_dims', 'activation_fn']


    def encoder(self, x, reuse=False):
        """ Encoder network.
        Args:
            :param x - input x.
            :param reuse - whether to reuse old network on create new one.
        Returns:
            A tuple z, softmax input logits
        """
        net = x

        with slim.arg_scope([slim.fully_connected],
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            activation_fn=self.activation_fn):
            for i, num_unit in enumerate(self.h_dims):
                net = slim.fully_connected(
                    net, num_unit,
                    normalizer_fn=slim.batch_norm,
                    reuse=reuse, scope='enc_{0}'.format(i))
                if self.dropout:
                    net = slim.dropout(net, keep_prob=self.keep_prob, is_training=self.is_training)

        # It is very important to batch normalize the output of encoder.
        z = slim.fully_connected(
            net, self.z_dim, activation_fn=None,
            normalizer_fn=slim.batch_norm,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            reuse=reuse, scope='enc_z')

        h = None

        # contrastive loss
  #      if self.enable_self:
       #     h = slim.fully_connected(
         #       z, self.rot_dim, activation_fn=None,
         #       normalizer_fn=slim.batch_norm,
          #      weights_initializer=tf.contrib.layers.xavier_initializer(),
           #    reuse=reuse, scope='ss_h')

        # This used when CE cost is enabled.
        if self.z_dim > 8:
            d_z = z[:, :-3]
            r_z = z[:, -3:]

        dim = self.y_dim

        if self.enable_cluster_loss:
            dim = self.n_clusters

        logits = slim.fully_connected(
            d_z, dim, activation_fn=None,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            scope='classifier_logits', reuse=reuse)

        rot_logits = None
        if self.z_dim > 8:
            rot_logits = slim.fully_connected(
                r_z, self.rot_dim, activation_fn=None,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                scope='rot_logits', reuse=reuse)
        else:
            rot_logits = slim.fully_connected(
                z, self.rot_dim, activation_fn=None,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                scope='rot_logits', reuse=reuse)
        return z, logits, rot_logits, h


    def decoder(self, z, reuse=False):
        """ Decoder Network.
        Args:
            :param z - latent variables z.
            :param reuse - whether to reuse old network on create new one.
        Returns:
            The reconstructed x
        """
        net = z

        with slim.arg_scope([slim.fully_connected],
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            activation_fn=self.activation_fn):
            h_dims_revese = [self.h_dims[i]
                             for i in range(len(self.h_dims) - 1, -1, -1)]
            for i, num_unit in enumerate(h_dims_revese):
                net = slim.fully_connected(
                    net, num_unit,
                    normalizer_fn=slim.batch_norm,
                    reuse=reuse, scope='dec_{0}'.format(i))
                if self.dropout:
                    net = slim.dropout(net, keep_prob=self.keep_prob, is_training=self.is_training)

        dec_out = slim.fully_connected(
            net, self.x_dim, activation_fn=tf.nn.tanh,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            reuse=reuse, scope='dec_out')
        return dec_out


    def build_model(self):
        self.x = tf.placeholder(tf.float32, shape=[None, self.x_dim])
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim])
        self.y = tf.placeholder(tf.float32, shape=[None, self.y_dim])
        self.rot = tf.placeholder(tf.float32, shape=[None, self.rot_dim])
        self.decouple = tf.placeholder(tf.float32, shape=[None, self.decouple_dim])
        self.d_z = None
        self.t_z = None

        self.z, logits, rot_logits, self.h = self.encoder(self.x)

        if self.enable_recon_loss:
            self.x_recon = self.decoder(self.z)
        else:
            self.x_recon = None
        # Calculate class mean
     #   self.trans_means = None
        self.trans_means = self.bucket_mean(self.z, tf.argmax(self.decouple, axis=1), self.n_clusters)
        self.class_means = self.bucket_mean(self.z, tf.argmax(self.y, axis=1), self.y_dim)
        if self.z_dim > 8:
            self.d_z = self.z[:, :-3]
            self.t_z = self.z[:, -3:]
            self.class_means = self.bucket_mean(self.d_z, tf.argmax(self.y, axis=1), self.y_dim)
            self.trans_means = self.bucket_mean(self.d_z, tf.argmax(self.decouple, axis=1), self.n_clusters)
        #   self.class_std = self.
        self.rot_means = self.bucket_mean(self.z, tf.argmax(self.rot, axis=1), self.rot_dim)
        self.loss_fn_training_op(self.x, self.y, self.z, self.h, self.decouple,
                                 logits, self.x_recon, self.class_means, self.rot, self.trans_means, rot_logits, self.d_z, self.t_z)
    #    self.loss_fn_training_op(self.x, self.y, self.z, self.h, logits, self.x_recon, self.class_means, self.rot)

        self.pred_prob = tf.nn.softmax(logits=logits)
   #     self.pred_prob_rot = tf.nn.softmax(logits=rot_logits)
        pred = tf.argmax(self.pred_prob, axis=1)
        actual = tf.argmax(self.y, axis=1)
        actual_rot = tf.argmax(self.rot, axis=1)
        self.acc = tf.reduce_mean(tf.cast(tf.equal(pred, actual), tf.float32))
        # For Inference, set is_training
        self.is_training = False
        self.z_test, logits_test, rot_logits_test, self.h_test = self.encoder(self.x)
        self.pred_prob_test = tf.nn.softmax(logits=logits_test)
        if self.enable_recon_loss:
            self.x_recon_test = self.decoder(self.z_test, reuse=True)
        self.is_training = True


class OpenNetCNN(OpenNetBase):
    """ OpenNet with convolutional and fully connected layers.
    Current supports simple architecture with alternating cov and pooling layers.
    """
    def __init__(self, x_dim, x_ch, y_dim, conv_units, hidden_units,
                 z_dim=6,
                 kernel_sizes=[5,5], strides=[1, 1], paddings='SAME',
                 pooling_enable=False, pooling_kernel=[2,2],
                 pooling_stride=[2,2], pooling_padding='SAME',
                 pooling_type='max', # 'avg' or 'max'
                 activation_fn=tf.nn.relu,

                 x_scale=UnitPosNegScale.scale,
                 x_inverse_scale=UnitPosNegScale.inverse_scale,
                 x_reshape=None,

                 opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9),
                 recon_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9),
                 c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9),
                 rot_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9),

                 dist='mean_separation_spread',
                 decision_dist_fn = 'euclidean',
                 threshold_type='global',
                 dropout = True, keep_prob=0.7,
                 batch_size=128, iterations=5000,
                 display_step=500, save_step=500,
                 model_directory=None,  # Directory to save trained model to.
                 density_estimation_factory=None, # deprecated
                 load_model_directory=None,
                 ce_loss=False, recon_loss=False, inter_loss=True, intra_loss=True,
                 div_loss=False,
                 decouple_loss=False,
                 trans_ce_loss=False,
                 intra_sample_loss=False,
                 cluster_loss=False,
                 mmf_extension=False,
                 tc_loss=False,
                 contamination=0.01,
                 mmf_comb = 0.0,
                 margin=8.0,  self_sup = False, n_rots=1, n_decouples=0, n_clusters=2
                 ):
        """
        Args:
        :param x_dim - dimension of the input
        :param y_dim - number of known classes.
        :param conv_units - a list of ints. The number of filters in each convolutional layer.
        :param hidden_units - a list of ints. The of units in each fully connected layer.
        :param z_dim - the number of latent variables.
        :param h_dims - an int or a list; number of units in the fully connected hidden layers of the
                        encoder network.
        :param kernel_sizes - a list or a list of lists. Size of the kernel of the conv2d.
                              If a list with two ints all layers use the same kernel size.
                              Otherwise if a list of list (example [[5,5], [4,4]]) each layer
                              will have different kernel size.
        :param strides - a list or a list of lists. The strides of each conv2d kernel.
        :param paddings - padding for each conv2d. Default 'SAME'.
        :param pooling_enable - if True, add pooling layer after each conv2d layer.
        :param pooling_kernel - a list or a list of lists. The size of the pooling kernel.
                                If a list with two ints all layers use the same kernel size.
                                Otherwise if a list of list (example [[5,5], [4,4]]) each layer
                                will have different kernel size.
        :param pooling_stride - a list or a list of lists. The strides of each pooing kernel.
        :param pooling_padding - padding for each pool2d layer. Default 'SAME'.
        :param pooling_type - pooling layer type. supported 'avg' or 'max'. Default max_pool2d.
        :param x_scale - an input scaling function. Default scale to range of [-1, 1].
                         If none, the input will not be scaled.
        :param x_inverse_scale - reverse scaling fn. by rescaling from [-1, 1] to original input scale.
                                 If None, the the output of decoder(if there is a decoder) will rescaled.
        :param x_reshape - a function to reshape the input before feeding to the networks input layer.
                            If None, the input will not be reshaped.
        :param opt - the Optimizer used when updating based on ii-loss.
                     Used when inter_loss and intra_loss are enabled. Default is AdamOptimizer.
        :param recon_opt - the Optimizer used when updating based on reconstruction-loss (Not used ii, ii+ce or ce).
                           Used when recon_loss is enabled. Default is AdamOptimizer.
        :param c_opt - the Optimizer used when updating based on cross entropy loss.
                       Used for ce and ii+ce modes (i.e. ce_loss is enabled). Default is AdamOptimizer.
        :param batch_size - training batch size.
        :param iterations - number of training iterations.
        :param display_step - training info displaying interval.
        :param save_step - model saving interval.
        :param model_directory - directory to save model in.
        :param dist - ii-loss calculation mode. Only 'mean_separation_spread' should be used.
        :param decision_dist_fn - outlier score distance functions
        :param threshold_type - outlier threshold mode. 'global' appears to give better results.
        :param ce_loss - Consider cross entropy loss. When enabled with intra_loss and inter_loss gives (ii+ce) mode.
        :param recon_loss - Experimental! Avoid enabling this.
        :param inter_loss - Consider inter-class separation. Should be enabled together with intra_loss for (ii-loss).
        :param intra_loss - Consider intra-class spread. Should be enabled together with inter_loss for (ii-loss).
        :param div_loss and combined_loss - Experimental. Avoid enabling them.
        :param contamination - contamination ratio used for outlier threshold estimation.
        """
        self.x_ch = x_ch

        # Conv layer config
        self.conv_units = conv_units
        if isinstance(kernel_sizes[0], list) or isinstance(kernel_sizes[0], tuple):
            assert len(conv_units) == len(kernel_sizes)
            self.kernel_sizes = kernel_sizes
        else:
            self.kernel_sizes = [kernel_sizes] * len(conv_units)

        if isinstance(strides[0], list) or isinstance(strides[0], tuple):
            assert len(conv_units) == len(strides)
            self.strides = strides
        else:
            self.strides = [strides] * len(conv_units)

        if isinstance(paddings, list):
            assert len(conv_units) == len(paddings)
            self.paddings = paddings
        else:
            self.paddings = [paddings] * len(conv_units)

        # Conv pooling config
        self.pooling_enable = pooling_enable
        assert pooling_type in ['avg', 'max']   # supported pooling types.
        self.pooling_type = pooling_type

        if isinstance(pooling_kernel[0], list) or isinstance(pooling_kernel[0], tuple):
            assert len(conv_units) == len(pooling_kernel)
            self.pooling_kernels = pooling_kernel
        else:
            self.pooling_kernels = [pooling_kernel] * len(conv_units)

        if isinstance(pooling_stride[0], list) or isinstance(pooling_stride[0], tuple):
            assert len(conv_units) == len(pooling_stride)
            self.pooling_strides = pooling_stride
        else:
            self.pooling_strides = [pooling_stride] * len(conv_units)

        if isinstance(pooling_padding, list):
            assert len(conv_units) == len(pooling_padding)
            self.pooling_paddings = pooling_padding
        else:
            self.pooling_paddings = [pooling_padding] * len(conv_units)

        # Fully connected layer config
        self.hidden_units = hidden_units

        self.activation_fn = activation_fn

        assert decision_dist_fn in ['euclidean', 'mahalanobis']

        super(OpenNetCNN, self).__init__(
            x_dim, y_dim, z_dim=z_dim,
            x_scale=x_scale, x_inverse_scale=x_inverse_scale, x_reshape=x_reshape,
            opt=opt, recon_opt=recon_opt, c_opt=c_opt, rot_opt=rot_opt, threshold_type=threshold_type,
            dist=dist, decision_dist_fn=decision_dist_fn, dropout=dropout, keep_prob=keep_prob,
            batch_size=batch_size, iterations=iterations,
            display_step=display_step, save_step=save_step,
            model_directory=model_directory, load_model_directory=load_model_directory,
            ce_loss=ce_loss, recon_loss=recon_loss, inter_loss=inter_loss, intra_loss=intra_loss,
            div_loss=div_loss, decouple_loss=decouple_loss,trans_ce_loss = trans_ce_loss, intra_sample_loss=intra_sample_loss,
            cluster_loss=cluster_loss, mmf_extension=mmf_extension, tc_loss=tc_loss, contamination=contamination,
            mmf_comb=mmf_comb, margin=margin,  self_sup=self_sup, n_rots=n_rots, n_decouples=n_decouples, n_clusters=n_clusters)


        self.model_params += ['x_ch', 'conv_units', 'kernel_sizes', 'strides', 'paddings',
                              'pooling_enable', 'pooling_type', 'pooling_kernel', 'pooling_strides',
                              'pooling_padding', 'hidden_units', 'activation_fn']


    def build_conv(self, x, reuse=False):
        """ Builds the convolutional layers.
        """
        net = x
        with slim.arg_scope([slim.conv2d], padding='SAME',
                            weights_initializer=tf.contrib.layers.xavier_initializer(),#tf.truncated_normal_initializer(stddev=0.01),
                            weights_regularizer=slim.l2_regularizer(0.0005),
                            activation_fn=self.activation_fn):
            for i, (c_unit, kernel_size, stride, padding, p_kernel, p_stride, p_padding) in enumerate(zip(
                    self.conv_units, self.kernel_sizes, self.strides, self.paddings,
                    self.pooling_kernels, self.pooling_strides, self.pooling_paddings)):
                # Conv
                net = slim.conv2d(net, c_unit, kernel_size, stride=stride,
                                  normalizer_fn=slim.batch_norm,
                                  reuse=reuse, padding=padding, scope='enc_conv{0}'.format(i))

                if self.display_step > 0:
                    print('Conv_{0}.shape = {1}'.format(i, net.get_shape()))
                # Pooling
                if self.pooling_enable:
                    if self.pooling_type == 'max':
                        net = slim.max_pool2d(net, kernel_size=p_kernel, scope='enc_pool{0}'.format(i),
                                              stride=p_stride, padding=p_padding)
                    elif self.pooling_type == 'avg':
                        net = slim.avg_pool2d(net, kernel_size=p_kernel, scope='enc_pool{0}'.format(i),
                                              stride=p_stride, padding=p_padding)

                    if self.display_step > 0:
                        print ('Pooling_{0}.shape = {1}'.format(i, net.get_shape()))
                # Dropout: Do NOT use dropout for conv layers. Experiments show it gives poor result.
        return net

    def encoder(self,  x, reuse=False):
        """ Builds the network.
        Args:
            :param x - input x.
            :param reuse - whether to reuse old network on create new one.
        Returns:
            Latent variables z and logits(which will be used if ce_loss is enabled.)
        """
        # Conv Layers
        net = self.build_conv(x, reuse=reuse)
        net = slim.flatten(net)

        # Fully Connected Layer
        with slim.arg_scope([slim.fully_connected], reuse=reuse,
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            weights_regularizer=slim.l2_regularizer(0.0005),
                            activation_fn=self.activation_fn):
            for i, h_unit in enumerate(self.hidden_units):
                net = slim.fully_connected(net, h_unit,
                                           normalizer_fn=slim.batch_norm,
                                           scope='enc_full{0}'.format(i))
                if self.dropout:
                    net = slim.dropout(net, keep_prob=self.keep_prob, is_training=self.is_training,
                                       scope='enc_full_dropout{0}'.format(i))

        # Latent Variable
        # It is very important to batch normalize the output of encoder.
        z = slim.fully_connected(
            net, self.z_dim, activation_fn=None,
            normalizer_fn=slim.batch_norm,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            reuse=reuse, scope='enc_z')

        h = None
    #    if self.enable_self:
     #       h = slim.fully_connected(
      #          z, self.rot_dim, activation_fn=None,
       #         normalizer_fn=slim.batch_norm,
        #        weights_initializer=tf.contrib.layers.xavier_initializer(),
         #       reuse=reuse, scope='ss_h')
        projection = None

        if self.z_dim > 8:
            d_z = z[:, :-3]
            t_z = z[:, -3:]

            logits = slim.fully_connected(
                d_z, self.y_dim, activation_fn=None,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                reuse=reuse, scope='classifier_logits')

            projection = slim.fully_connected(
                d_z, self.y_dim, activation_fn=None,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                reuse=reuse, scope='projection')


            rot_logits = None
            if self.enable_trans_ce_loss:
                rot_logits = slim.fully_connected(
                    t_z, 4, activation_fn=None,
                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                    reuse=reuse, scope='rot_logits')

        else:
            logits = slim.fully_connected(
                z, self.y_dim, activation_fn=None,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                reuse=reuse, scope='classifier_logits')


            rot_logits = None
            if self.enable_self:
                rot_logits = slim.fully_connected(
                    z, 4, activation_fn=None,
                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                    reuse=reuse, scope='rot_logits')

        return z, logits, rot_logits, h, projection

    def decoupling_decoder(self, z, reuse=False):
        """ Decoder Network. Experimental and not complete.
        Args:
            :param z - latent variables z.
            :param reuse - whether to reuse old network on create new one.
        Returns:
            The reconstructed x
        """
        net = z
        with slim.arg_scope([slim.fully_connected],
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            activation_fn=self.activation_fn):
            h_dims_revese = [self.hidden_units[i]
                             for i in range(len(self.hidden_units) - 1, -1, -1)]
            for i, num_unit in enumerate(h_dims_revese):
                net = slim.fully_connected(
                    net, num_unit,
                    normalizer_fn=slim.batch_norm,
                    reuse=reuse, scope='decouple_{0}'.format(i))
                if self.dropout:
                    net = slim.dropout(net, keep_prob=self.keep_prob, is_training=self.is_training)

        dec_out = slim.fully_connected(
            net, self.x_dim[0] * self.x_dim[1] * self.x_ch, activation_fn=tf.nn.tanh,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            reuse=reuse, scope='decouple_out')
        return dec_out

    def decoder(self, z, reuse=False):
        """ Decoder Network. Experimental and not complete.
        Args:
            :param z - latent variables z.
            :param reuse - whether to reuse old network on create new one.
        Returns:
            The reconstructed x
        """
        print("decoder here")
        net = z
        if self.z_dim > 8:
            net = z[:, :-3]

        with slim.arg_scope([slim.fully_connected],
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            activation_fn=self.activation_fn):
            h_dims_revese = [self.hidden_units[i]
                             for i in range(len(self.hidden_units) - 1, -1, -1)]
            for i, num_unit in enumerate(h_dims_revese):
                net = slim.fully_connected(
                    net, num_unit,
                    normalizer_fn=slim.batch_norm,
                    reuse=reuse, scope='dec_{0}'.format(i))
                if self.dropout:
                    net = slim.dropout(net, keep_prob=self.keep_prob, is_training=self.is_training)

        dec_out = slim.fully_connected(
            net, self.x_dim[0] * self.x_dim[1] * self.x_ch, activation_fn=tf.nn.tanh,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            reuse=reuse, scope='dec_out')
        return dec_out


    def build_model(self):
        """ Builds the network graph.
        """
        self.x = tf.placeholder(tf.float32, [None, self.x_dim[0], self.x_dim[1], self.x_ch])
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim])
        self.d_z = tf.placeholder(tf.float32, shape=[None, self.z_dim-3])
        self.t_z = tf.placeholder(tf.float32, shape=[None, 3])
        self.projection = tf.placeholder(tf.float32, shape=[None, self.y_dim])
        self.y = tf.placeholder(tf.float32, shape=[None, self.y_dim])
        self.decouple = tf.placeholder(tf.float32, shape=[None, self.decouple_dim])

        self.x_unknown = tf.placeholder(tf.float32, [None, self.x_dim[0], self.x_dim[1], self.x_ch])
        self.z_unknown = tf.placeholder(tf.float32, shape=[None, self.z_dim])
        self.d_z_unknown = tf.placeholder(tf.float32, shape=[None, self.z_dim - 3])
        self.t_z_unknown = tf.placeholder(tf.float32, shape=[None, 3])
        self.y_unknown = tf.placeholder(tf.float32, shape=[None, self.y_dim])
        self.x_recon = tf.placeholder(tf.float32, [None, self.x_dim[0], self.x_dim[1], self.x_ch])
        self.rot = tf.placeholder(tf.float32, shape=[None, self.x_dim[0], self.x_dim[1], self.x_ch])

        self.center_unknown = tf.placeholder(tf.float32, shape=[self.n_clusters, self.z_dim])

        if self.z_dim > 8:
            self.center_unknown = tf.placeholder(tf.float32, shape=[self.n_clusters, self.z_dim-3])

     #
        if type(self.rot_dim) is int:
            self.rot_unknown = tf.placeholder(tf.float32, shape=[None, self.rot_dim])
        else:
            self.rot_unknown = tf.placeholder(tf.float32, shape=[None, self.x_dim[0], self.x_dim[1], self.x_ch])

        self.z, logits, rot_logits, self.h, self.projection = self.encoder(self.x)

        unknown_logits=None
        if self.enable_cluster_loss:
            self.z_unknown, unknown_logits, _, _, _ = self.encoder(self.x_unknown, reuse=True)

        if self.enable_decouple:
            self.x_decouple_recon = self.decoupling_decoder(self.z)
            print("enable decouple")
        else:
            self.x_decouple_recon = None
            print("disable decouple")

        # Calculate class mean
        self.trans_means = self.bucket_mean(self.z, tf.argmax(self.decouple, axis=1), 4)
        self.class_means = self.bucket_mean(self.z, tf.argmax(self.y, axis=1), self.y_dim)

        if self.z_dim > 8:
            self.d_z = self.z[:, :-3]
            self.t_z = self.z[:, -3:]
            self.class_means = self.bucket_mean(self.d_z, tf.argmax(self.y, axis=1), self.y_dim)
            self.trans_means = self.bucket_mean(self.d_z, tf.argmax(self.decouple, axis=1), 4)
            if self.enable_cluster_loss:
                self.d_z_unknown = self.z_unknown[:, :-3]
                self.t_z_unknown = self.z_unknown[:, -3:]

        if self.enable_recon_loss:
          #  print("z shape: ", self.z.shape)
            self.x_recon = self.decoder(self.z)

        #    if self.enable_self:
     #       self.trans_means = self.bucket_mean(self.z, tf.argmax(self.rot, axis=1), self.rot_dim)
      #  self.trans_means = self.z[:self.rot_dim]

        # print("z shape")
        # print(self.z.shape)
        # print("y shape")
        # print(self.y.shape)
        # print("class mean shape")
        # print(self.class_means.shape)
        # print("rot shape")
        # print(self.rot.shape)
        # print("trans mean shape")
        # print(self.trans_means.shape)
        self.loss_fn_training_op(slim.flatten(self.x), self.y, self.z, self.h, self.decouple,
                                 logits, self.x_recon, self.x_decouple_recon, self.class_means, slim.flatten(self.rot),
                                 self.trans_means, rot_logits, self.d_z, self.t_z, self.projection, slim.flatten(self.x_unknown),
                                 self.y_unknown, self.z_unknown, self.d_z_unknown, self.center_unknown, unknown_logits=unknown_logits)


        self.pred_prob = tf.nn.softmax(logits=logits)
        pred = tf.argmax(self.pred_prob, axis=1)
        actual = tf.argmax(self.y, axis=1)
        self.acc = tf.reduce_mean(tf.cast(tf.equal(pred, actual), tf.float32))
      #  print("self.c: " + str(self.acc))
        # For Inference, set is_training. Can be done in a better, this should do for now.
        self.is_training = False
        self.z_test, logits_test, _, self.h_test, _ = self.encoder(self.x, reuse=True)
        self.pred_prob_test = tf.nn.softmax(logits=logits_test)
        if self.enable_recon_loss:
            self.x_recon_test = self.decoder(self.z_test, reuse=True)
        self.is_training = True

