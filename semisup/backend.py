"""
Copyright 2016 Google Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Utility functions for Association-based semisupervised training.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
import tensorflow.contrib.slim as slim


def show_sample_img(img):
  import matplotlib.pyplot as plt
  plt.imshow(img.reshape(28, 28), cmap='gray')
  plt.show()

def show_sample_img_inline(imgs):
  import matplotlib.pyplot as plt
  f, axarr = plt.subplots(1, max(len(imgs),2))
  for ind, img in enumerate(imgs):
    axarr[ind].imshow(img.reshape(28, 28), cmap='gray')
  plt.show()


def create_input(input_images, input_labels, batch_size, shuffle=False):
  """Create preloaded data batch inputs.

  Args:
    input_images: 4D numpy array of input images.
    input_labels: 2D numpy array of labels.
    batch_size: Size of batches that will be produced.

  Returns:
    A list containing the images and labels batches.
  """
  if batch_size == -1:
    batch_size = input_labels.shape[0]
  if input_labels is not None:
    image, label = tf.train.slice_input_producer([input_images, input_labels])
    if shuffle:
      return tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=500, min_after_dequeue=100)
    else:
      return tf.train.batch([image, label], batch_size=batch_size)
  else:
    image = tf.train.slice_input_producer([input_images])
    return tf.train.batch(image, batch_size=batch_size)


def create_per_class_inputs(image_by_class, n_per_class, class_labels=None):
  """Create batch inputs with specified number of samples per class.

  Args:
    image_by_class: List of image arrays, where image_by_class[i] containts
        images sampled from the class class_labels[i].
    n_per_class: Number of samples per class in the output batch.
    class_labels: List of class labels. Equals to range(len(image_by_class)) if
        not provided.

  Returns:
    images: Tensor of n_per_class*len(image_by_class) images.
    labels: Tensor of same number of labels.
  """
  if class_labels is None:
    class_labels = np.arange(len(image_by_class))
  batch_images, batch_labels = [], []
  for images, label in zip(image_by_class, class_labels):
    labels = tf.fill([len(images)], label)
    images, labels = create_input(images, labels, n_per_class)
    batch_images.append(images)
    batch_labels.append(labels)
  return tf.concat(batch_images, 0), tf.concat(batch_labels, 0)


def create_per_class_inputs_sub_batch(image_by_class, n_per_class, class_labels=None):
  """Create batch inputs with specified number of samples per class.

  Args:
    image_by_class: List of image arrays, where image_by_class[i] containts
        images sampled from the class class_labels[i].
    n_per_class: Number of samples per class in the output batch.
    class_labels: List of class labels. Equals to range(len(image_by_class)) if
        not provided.

  Returns:
    images: Tensor of n_per_class*len(image_by_class) images.
    labels: Tensor of same number of labels.
  """
  batch_images, batch_labels = [], []
  if class_labels is None:
    class_labels = np.arange(len(image_by_class))

  for images, label in zip(image_by_class, class_labels):
    labels = np.ones(len(images)) * label

    indices = np.random.choice(range(0, len(images)), n_per_class)
    # todo make sure we don't miss a sample here

    batch_images.extend(images[indices])
    batch_labels.extend(labels[indices])

  batch_images = np.asarray(batch_images)
  batch_labels = np.asarray(batch_labels, np.int)

  imgs, lbls = create_input(batch_images, batch_labels, batch_size=20, shuffle=True)

  return imgs, lbls


def sample_by_label(images, labels, n_per_label, num_labels, seed=None):
  """Extract equal number of sampels per class."""
  res = []
  rng = np.random.RandomState(seed=seed)
  for i in range(num_labels):
    a = images[labels == i]
    if n_per_label == -1:  # use all available labeled data
      res.append(a)
    else:  # use randomly chosen subset
      inds = rng.choice(len(a), n_per_label, False)
      res.append(a[inds])
  return res


def create_virt_emb(n, size):
  """Create virtual embeddings."""
  emb = slim.variables.model_variable(
      name='virt_emb',
      shape=[n, size],
      dtype=tf.float32,
      trainable=True,
      initializer=tf.random_normal_initializer(stddev=0.01))
  return emb


def confusion_matrix(labels, predictions, num_labels):
  """Compute the confusion matrix."""
  rows = []
  for i in range(num_labels):
    row = np.bincount(predictions[labels == i], minlength=num_labels)
    rows.append(row)
  return np.vstack(rows)


def softmax(x):
  maxes = np.amax(x, axis=1)
  maxes = maxes.reshape(maxes.shape[0], 1)
  e = np.exp(x - maxes)
  dist = e / np.sum(e, axis=1).reshape(maxes.shape[0], 1)
  return dist

def one_hot(a, depth):
  b = np.zeros((a.size, depth))
  b[np.arange(a.size), a] = 1
  return b

def logistic_growth(current_step, target, steps):
  assert target > 0., 'Target value must be positive.'
  alpha = 5. / steps
  return target * (np.tanh(alpha * (current_step - steps / 2.)) + 1.) / 2.

def apply_envelope(type, step, final_weight, growing_steps, delay):
  assert growing_steps > 0, "Growing steps for envelope must be > 0."
  step = step - delay
  final_step = growing_steps + delay

  if type is None:
    value = final_weight

  elif type in ['sigmoid', 'sigmoidal', 'logistic', 'log']:
    value = logistic_growth(step, final_weight, final_step)

  elif type in ['linear', 'lin']:
    m = float(final_weight) / (
      growing_steps) if not growing_steps == 0.0 else 999.
    value = m * step
  else:
    raise NameError('Invalid type: ' + str(type))

  return np.clip(value, 0., final_weight)

from tensorflow.python.framework import ops
LOSSES_COLLECTION = ops.GraphKeys.LOSSES


def l1_loss(tensor, weight, scope=None):
  """Define a L1Loss, useful for regularize, i.e. lasso.
  Args:
    tensor: tensor to regularize.
    weight: tensor: scale the loss by this factor.
    scope: Optional scope for name_scope.
  Returns:
    the L1 loss op.
  """
  with tf.name_scope(scope, 'L1Loss', [tensor]):
      #weight = tf.convert_to_tensor(weight,
      #                              dtype=tensor.dtype.base_dtype,
      #                              name='loss_weight')
      loss = tf.multiply(weight, tf.reduce_sum(tf.abs(tensor)), name='value')
      tf.add_to_collection(LOSSES_COLLECTION, loss)
  return loss



class SemisupModel(object):
  """Helper class for setting up semi-supervised training."""

  def __init__(self, model_func, num_labels, input_shape, test_in=None,
               optimizer='adam', emb_size=128, dropout_keep_prob=1, augmentation_function=None):
    """Initialize SemisupModel class.

    Creates an evaluation graph for the provided model_func.

    Args:
      model_func: Model function. It should receive a tensor of images as
          the first argument, along with the 'is_training' flag.
      num_labels: Number of taget classes.
      input_shape: List, containing input images shape in form
          [height, width, channel_num].
      test_in: None or a tensor holding test images. If None, a placeholder will
        be created.
    """

    self.num_labels = num_labels
    self.step = slim.get_or_create_global_step()
    self.ema = tf.train.ExponentialMovingAverage(0.99, self.step)
    self.emb_size = emb_size

    self.test_batch_size = 100

    self.model_func = model_func
    self.augmentation_function = augmentation_function
    self.optimizer = optimizer
    self.dropout_keep_prob = dropout_keep_prob

    if test_in is not None:
      self.test_in = test_in
    else:
      self.test_in = tf.placeholder(np.float32, [None] + input_shape, 'test_in')

    self.test_emb = self.image_to_embedding(self.test_in, is_training=False)
    self.test_logit = self.embedding_to_logit(self.test_emb, is_training=False)

  def reset_optimizer(self, sess):
    optimizer_slots = [
      self.trainer.get_slot(var, name)
      for name in self.trainer.get_slot_names()
      for var in tf.model_variables()
    ]
    if isinstance(self.trainer, tf.train.AdamOptimizer):
      optimizer_slots.extend([
        self.trainer._beta1_power, self.trainer._beta2_power
      ])
    init_op = tf.variables_initializer(optimizer_slots)
    sess.run(init_op)

  def image_to_embedding(self, images, is_training=True):
    """Create a graph, transforming images into embedding vectors."""
    with tf.variable_scope('net', reuse=is_training):
      self.model = self.model_func(images, is_training=is_training, emb_size=self.emb_size,
                                   dropout_keep_prob=self.dropout_keep_prob, augmentation_function=self.augmentation_function)
      return self.model

  def embedding_to_logit(self, embedding, is_training=True):
    """Create a graph, transforming embedding vectors to logit classs scores."""
    with tf.variable_scope('net', reuse=is_training):
      return slim.fully_connected(
          embedding,
          self.num_labels,
          activation_fn=None,
          weights_regularizer=slim.l2_regularizer(1e-4))

  def add_semisup_loss(self, a, b, labels, walker_weight=1.0, visit_weight=1.0, proximity_weight=None, normalize_along_classes=False):
    """Add semi-supervised classification loss to the model.

    The loss consists of two terms: "walker" and "visit".

    Args:
      a: [N, emb_size] tensor with supervised embedding vectors.
      b: [M, emb_size] tensor with unsupervised embedding vectors.
      labels : [N] tensor with labels for supervised embeddings.
      walker_weight: Weight coefficient of the "walker" loss.
      visit_weight: Weight coefficient of the "visit" loss.
    """

    equality_matrix = tf.equal(tf.reshape(labels, [-1, 1]), labels)
    equality_matrix = tf.cast(equality_matrix, tf.float32)
    p_target = (equality_matrix / tf.reduce_sum(
        equality_matrix, [1], keep_dims=True))#*2  # TODO why does this help??

    match_ab = tf.matmul(a, b, transpose_b=True, name='match_ab')
    p_ab = tf.nn.softmax(match_ab, name='p_ab')
    p_ba = tf.nn.softmax(tf.transpose(match_ab), name='p_ba')
    p_aba = tf.matmul(p_ab, p_ba, name='p_aba')

    #self.create_walk_statistics(p_aba, equality_matrix)
    
    loss_aba = tf.losses.softmax_cross_entropy(
        p_target,
        tf.log(1e-8 + p_aba),
        weights=walker_weight,
        scope='loss_aba')

    if normalize_along_classes:
      self.add_visit_loss_class_normalized(p_ab, p_target, visit_weight)
    else:
     self.add_visit_loss(p_ab, visit_weight)

    if proximity_weight is not None:
      self.add_visit_loss_bab(p_ab, p_ba, proximity_weight)

    tf.summary.scalar('Loss_aba', loss_aba)

  def add_visit_loss(self, p, weight=1.0):
    """Add the "visit" loss to the model.

    Args:
      p: [N, M] tensor. Each row must be a valid probability distribution
          (i.e. sum to 1.0)
      weight: Loss weight.
    """
    visit_probability = tf.reduce_mean(
        p, [0], keep_dims=True, name='visit_prob')
    t_nb = tf.shape(p)[1]
    visit_loss = tf.losses.softmax_cross_entropy(
        tf.fill([1, t_nb], 1.0 / tf.cast(t_nb, tf.float32)),
        tf.log(1e-8 + visit_probability),
        weights=weight,
        scope='loss_visit')

    tf.summary.scalar('Loss_Visit', visit_loss)

  def add_visit_loss_class_normalized(self, p, p_target, weight=1.0):
    """Add the "visit" loss to the model.

    Args:
      p: [N, M] tensor. Each row must be a valid probability distribution
          (i.e. sum to 1.0)
      p_target [N] tensor. See p_target from semisup_loss
      weight: Loss weight.
    """

    scale_f = tf.diag_part(p_target)
    p_norm = tf.transpose(tf.multiply(tf.transpose(p), scale_f))

    visit_probability = tf.reduce_sum(
      p_norm, [0], keep_dims=True, name='visit_prob')
    visit_probability = visit_probability * (1 / tf.reduce_sum(visit_probability))

    # compare with old one
    visit_probability_old = tf.reduce_mean(
      p, [0], keep_dims=True, name='visit_prob')

    self.dif = tf.reduce_sum(visit_probability - visit_probability_old)
    self.vp = visit_probability
    self.vpo = visit_probability_old

    t_nb = tf.shape(p)[1]
    visit_loss = tf.losses.softmax_cross_entropy(
        tf.fill([1, t_nb], 1.0 / tf.cast(t_nb, tf.float32)),
        tf.log(1e-8 + visit_probability),
        weights=weight,
        scope='loss_visit')

    tf.summary.scalar('Loss_Visit', visit_loss)

  def add_visit_loss_bab(self, p_ab, p_ba, weight=1.0):
    """Add the "visit" loss to the model.

    Args:
      p_ab, p_ba
      weight: Loss weight.
    """
    p_bab = tf.matmul(p_ba, p_ab, name='p_bab')

    visit_probability = tf.reduce_mean(p_bab, [0], name='visit_prob_bab', keep_dims=True)

    t_nb = tf.shape(p_bab)[1]
    visit_loss = tf.losses.softmax_cross_entropy(
        tf.fill([1, t_nb], 1.0 / tf.cast(t_nb, tf.float32)),
        tf.log(1e-8 + visit_probability),
        weights=weight,
        scope='loss_visit_bab')

    tf.summary.scalar('Loss_Visit_bab', visit_loss)

  def add_logit_loss(self, logits, labels, weight=1.0, smoothing=0.0):
    """Add supervised classification loss to the model."""

    logit_loss = tf.losses.softmax_cross_entropy(
        tf.one_hot(labels, logits.get_shape()[-1]),
        logits,
        scope='loss_logit',
        weights=weight,
        label_smoothing=smoothing)

    tf.summary.scalar('Loss_Logit', logit_loss)

  def create_walk_statistics(self, p_aba, equality_matrix):
    """Adds "walker" loss statistics to the graph.

    Args:
      p_aba: [N, N] matrix, where element [i, j] corresponds to the
          probalility of the round-trip between supervised samples i and j.
          Sum of each row of 'p_aba' must be equal to one.
      equality_matrix: [N, N] boolean matrix, [i,j] is True, when samples
          i and j belong to the same class.
    """
    # Using the square root of the correct round trip probalilty as an estimate
    # of the current classifier accuracy.
    per_row_accuracy = 1.0 - tf.reduce_sum((equality_matrix * p_aba), 1)**0.5
    estimate_error = tf.reduce_mean(
        1.0 - per_row_accuracy, name=p_aba.name[:-2] + '_esterr')
    self.add_average(estimate_error)
    self.add_average(p_aba)

    tf.summary.scalar('Stats_EstError', estimate_error)

  def add_average(self, variable):
    """Add moving average variable to the model."""
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, self.ema.apply([variable]))
    average_variable = tf.identity(
        self.ema.average(variable), name=variable.name[:-2] + '_avg')
    return average_variable

  def add_emb_regularization(self, embs, weight):
    """weight should be a tensor"""
    l1_loss(embs, weight)

  def create_train_op(self, learning_rate):
    """Create and return training operation."""

    slim.model_analyzer.analyze_vars(
        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), print_info=True)

    print(tf.losses.get_losses())

    self.train_loss = tf.losses.get_total_loss()
    self.train_loss_average = self.add_average(self.train_loss)

    tf.summary.scalar('Learning_Rate', learning_rate)
    tf.summary.scalar('Loss_Total_Avg', self.train_loss_average)
    tf.summary.scalar('Loss_Total', self.train_loss)

    if self.optimizer == 'sgd':
      self.trainer = tf.train.MomentumOptimizer(
            learning_rate, 0.9, use_nesterov=False)
    elif self.optimizer == 'adam':
      self.trainer = tf.train.AdamOptimizer(learning_rate)
    else:
      print('unrecognized optimizer')

    self.train_op = slim.learning.create_train_op(self.train_loss, self.trainer)
    return self.train_op

  def calc_embedding(self, images, endpoint, sess):
    """Evaluate 'endpoint' tensor for all 'images' using batches."""
    batch_size = self.test_batch_size
    emb = []
    for i in range(0, len(images), batch_size):
      emb.append(endpoint.eval({self.test_in: images[i: i+batch_size]}, session=sess))
    return np.concatenate(emb)

  def classify(self, images, session):
    """Compute logit scores for provided images."""
    return self.calc_embedding(images, self.test_logit, session)

  def get_images(self, img_queue, lbl_queue, num_batches, sess):
    imgs = []
    lbls = []

    for i in range(int(num_batches)):
      i_, l_ = sess.run([img_queue, lbl_queue])
      imgs.append(i_)
      lbls.append(l_)

    images = np.vstack(imgs)
    labels = np.hstack(lbls)

    return images, labels

  def classify_using_embeddings(self, sup_imgs, sup_lbls, test_images, test_labels, sess):

    if sup_imgs.shape:  # convert tensor to array
      sup_imgs, sup_lbls = self.get_images(sup_imgs, sup_lbls, 1, sess)

    sup_embs = self.calc_embedding(sup_imgs, self.test_emb, sess)
    test_embs = self.calc_embedding(test_images, self.test_emb, sess)

    match_ab = np.dot(sup_embs, np.transpose(test_embs))
    p_ba = softmax(np.transpose(match_ab))

    pred_ids = np.dot(p_ba, one_hot(sup_lbls, depth=self.num_labels))
    preds = np.argmax(pred_ids, axis=1)

    return np.mean(preds == test_labels)

  def propose_samples(self, sup_imgs, sup_lbls, train_images, train_labels, sess, n_samples=1, vis=False):
    sup_embs = self.calc_embedding(sup_imgs, self.test_emb, sess)
    train_embs = self.calc_embedding(train_images, self.test_emb, sess)

    match_ab = np.dot(sup_embs, np.transpose(train_embs))
    p_ba = softmax(np.transpose(match_ab))

    # add values from a that share class
    preds = np.dot(p_ba, one_hot(np.asarray(sup_lbls, np.int64), depth=self.num_labels))

    # calculate sample confidence: sample confidence + confidence of close-by_samples
    sample_conf = np.var(preds, axis=1)
    conf_thresh = np.percentile(sample_conf, 5)
    print('conf threshold:', conf_thresh)
    unconf_sample_indices = np.where(sample_conf < conf_thresh)[0][:10000]
    print(unconf_sample_indices.shape)
    #todo is empty
    unconf_train_embs = train_embs[unconf_sample_indices]

    # distances to other unlabeled samples
    # not normalized
    p_bb = np.dot(unconf_train_embs, np.transpose(unconf_train_embs))
    p_bb_or = p_bb.copy()
    # ignore faraway samples
    print('p', np.percentile(p_bb, 90))
    p_bb[p_bb < np.percentile(p_bb, 90)] = 0
    p_bb[p_bb > 1] = 1

    # add up the 'inconfidence' of all close samples
    region_conf = np.dot(p_bb, conf_thresh - sample_conf[unconf_sample_indices])

    # indices = np.argpartition(region_conf, kth=n_samples)[:n_samples]
    indices = np.argsort(-region_conf)[:n_samples]  # sort descending
    or_indices = unconf_sample_indices[indices]

    if vis:
      for i in or_indices:
          p_bb_ind = np.where(unconf_sample_indices == i)[0]
          show_sample_img(train_images[i])
          print('existing samples in training set from same class')
          inds = np.where(sup_lbls == train_labels[i])[0]
          imgs = sup_imgs[inds]
          show_sample_img_inline(imgs)

          print(preds[i, :])
          print(p_ba[i, :])
          print('close, also unconfident samples')
          uinds = np.argsort(-p_bb_or[p_bb_ind,:])[0,:10]
          orinds = unconf_sample_indices[uinds]
          show_sample_img_inline(train_images[orinds])

    return or_indices

  def propose_samples_pb(self, sup_imgs, sup_lbls, train_images, train_labels, sess, n_samples=1, vis=False):
    sup_embs = self.calc_embedding(sup_imgs, self.test_emb, sess)
    train_embs = self.calc_embedding(train_images, self.test_emb, sess)

    match_ab = np.dot(sup_embs, np.transpose(train_embs))
    p_ba = softmax(np.transpose(match_ab))

    preds = np.dot(p_ba, one_hot(np.asarray(sup_lbls, np.int64), depth=self.num_labels))

    p_ab = softmax(match_ab)
    sample_conf = np.mean(p_ab, axis=0)   # p_b

    conf_thresh = np.percentile(sample_conf, 15)
    m_conf_thresh = np.percentile(sample_conf, 1)

    print('confidence thresholds', conf_thresh, m_conf_thresh)
    unconf_sample_indices = np.where((m_conf_thresh < sample_conf) & (sample_conf < conf_thresh))[0][:10000]
    unconf_train_embs = train_embs[unconf_sample_indices]

    print(np.bincount(train_labels[unconf_sample_indices]))

    row_sums = np.square(unconf_train_embs).sum(axis=1)
    unconf_train_embs = unconf_train_embs / np.sqrt(row_sums[:, np.newaxis])

    # distances to other unlabeled samples
    # not normalized
    p_bb = np.dot(unconf_train_embs, np.transpose(unconf_train_embs))

    p_bb1 = p_bb.copy()
    p_bb1[p_bb1 < np.percentile(p_bb1, 20)] = 0

    # low score on normalized p_bb1 -> many close neighbours
    p_bb1 = softmax(p_bb1)
    close_neighbour_score = np.diag(p_bb1)  # lower is better

    th2 = np.percentile(close_neighbour_score, 10)
    sm = np.where(close_neighbour_score < th2)[0]

    # --> we should not choose lowest sample conf
    print(train_labels[np.argsort(sample_conf[unconf_sample_indices[sm]])[:20]])

    print(np.bincount(train_labels[unconf_sample_indices[sm]]))
    # indices = np.argpartition(region_conf, kth=n_samples)[:n_samples]
    indices = np.argsort(close_neighbour_score)[:n_samples]  # sort descending
    or_indices = unconf_sample_indices[indices]

    if vis:
      for count, i in enumerate(or_indices):
        index = indices[count]
        p_bb_ind = np.where(unconf_sample_indices == i)[0]
        show_sample_img(train_images[i])
        print('existing samples in training set from same class')
        inds = np.where(sup_lbls == train_labels[i])
        show_sample_img_inline(sup_imgs[inds])

        print('confidence', preds[i, :], np.var(preds[i, :]), sample_conf[i], 'score', close_neighbour_score[index])

        print('close, also unconfident samples')
        uinds = np.argsort(-p_bb[p_bb_ind, :])[0, :10]
        orinds = unconf_sample_indices[uinds]
        show_sample_img_inline(train_images[orinds])

    return or_indices

  def propose_samples_pb_sampling(self, sup_imgs, sup_lbls, train_images, train_labels, sess, n_samples=1, vis=False):
    sup_embs = self.calc_embedding(sup_imgs, self.test_emb, sess)
    train_embs = self.calc_embedding(train_images, self.test_emb, sess)

    match_ab = np.dot(sup_embs, np.transpose(train_embs))
    p_ba = softmax(np.transpose(match_ab))

    preds = np.dot(p_ba, one_hot(np.asarray(sup_lbls, np.int64), depth=self.num_labels))

    p_ab = softmax(match_ab)
    sample_conf = np.mean(p_ab, axis=0)   # p_b

    conf_thresh = np.percentile(sample_conf, 15)
    m_conf_thresh = np.percentile(sample_conf, 1)

    print('confidence thresholds', conf_thresh, m_conf_thresh)
    unconf_sample_indices = np.where((m_conf_thresh < sample_conf) & (sample_conf < conf_thresh))[0][:10000]
    unconf_train_embs = train_embs[unconf_sample_indices]

    print('class distribution for not confident samples', np.bincount(train_labels[unconf_sample_indices]))

    row_sums = np.square(unconf_train_embs).sum(axis=1)
    unconf_train_embs = unconf_train_embs / np.sqrt(row_sums[:, np.newaxis])

    # distances to other unlabeled samples
    # not normalized
    p_bb = np.dot(unconf_train_embs, np.transpose(unconf_train_embs))

    p_bb1 = p_bb.copy()
    p_bb1[p_bb1 < np.percentile(p_bb1, 20)] = 0

    # low score on normalized p_bb1 -> many close neighbours
    p_bb1 = softmax(p_bb1)
    close_neighbour_score = np.diag(p_bb1)  # lower is better

    th2 = np.percentile(close_neighbour_score, 10)
    sm = np.where(close_neighbour_score < th2)[0]

    print('class distribution for not confident and "clustered" samples', np.bincount(train_labels[unconf_sample_indices[sm]]))

    or_indices = unconf_sample_indices[sm[np.random.choice(len(sm), n_samples, replace=False)]]

    if vis:
      for count, i in enumerate(or_indices):
        p_bb_ind = np.where(unconf_sample_indices == i)[0]
        show_sample_img(train_images[i])
        print('existing samples in training set from same class')
        inds = np.where(sup_lbls == train_labels[i])
        show_sample_img_inline(sup_imgs[inds])

        print('confidence', preds[i, :], np.var(preds[i, :]), sample_conf[i])

        print('close, also unconfident samples')
        uinds = np.argsort(-p_bb[p_bb_ind, :])[0, :10]
        orinds = unconf_sample_indices[uinds]
        show_sample_img_inline(train_images[orinds])

    return or_indices

  def propose_samples_random(self, sup_imgs, sup_lbls, train_images, train_labels, sess, n_samples=1, vis=False):
    rng = np.random.RandomState()
    indices = rng.choice(len(train_images), n_samples, False)

    if vis:
      for i in indices:
          show_sample_img(train_images[i])
          print('existing samples in training set from same class')
          inds = np.where(sup_lbls == train_labels[i])
          show_sample_img_inline(sup_imgs[inds])
    return indices


  def propose_samples_min_var_logit(self, sup_imgs, sup_lbls, train_images, train_labels, sess, n_samples=1, vis=False):
    embs = self.calc_embedding(train_images, self.test_logit, sess)

    var = np.var(embs, axis = 1)
    indices = np.argpartition(var, kth=n_samples)[:n_samples]
    if vis:
      for i in indices:
          show_sample_img(train_images[i])
          print('existing samples in training set from same class')
          inds = np.where(sup_lbls == train_labels[i])
          show_sample_img_inline(sup_imgs[inds])

    print('indices', indices, var[indices])
    return indices


  def propose_samples_random_classes(self, sup_imgs, sup_lbls, train_images, train_labels, sess, n_samples=1, vis=False):
    """
    propose samples randomly, but ensure that the class distribution stays equal
    this then approaches training with more samples
    in the context of active learning should be considered 'cheating', as we use the class label info
     of all training samples to pick a sample
    """
    label_count = np.bincount(np.asarray(sup_lbls, np.int64), minlength=self.num_labels)

    target_per_class = 4
    p = np.ones(self.num_labels) * target_per_class - label_count
    p = p / np.sum(p)

    print('p',p)
    rng = np.random.RandomState()
    classes = rng.choice(self.num_labels, n_samples, False, p)

    indices = []
    for i in classes:
      inds_from_class = np.where(train_labels == i)[0]
      print(rng.choice(inds_from_class, 1, False))
      indices = indices + [rng.choice(inds_from_class, 1, False)[0]]

    if vis:
      for i in indices:
          show_sample_img(train_images[i])
          print('existing samples in training set from same class')
          inds = np.where(sup_lbls == train_labels[i])
          show_sample_img_inline(sup_imgs[inds])
    return indices

