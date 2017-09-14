#! /usr/bin/env python
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

Active learning on mnist, starting with one sample per class

With following parameters, should reach around 1-1.5% error with 50 samples (20 given, 30 added)

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)
import semisup

from tensorflow.python.platform import app
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

if __name__ == '__main__':
  flags.DEFINE_integer('sup_per_class', 2,
                       'Number of labeled samples used per class.')

  flags.DEFINE_integer('sup_seed', -1,
                       'Integer random seed used for labeled set selection.')

  flags.DEFINE_integer('sup_per_batch', -1,  # -1: use all that we have
                       'Number of labeled samples per class per batch.')

  flags.DEFINE_integer('unsup_batch_size', 30,
                       'Number of unlabeled samples per batch.')

  flags.DEFINE_integer('eval_interval', 500,
                       'Number of steps between evaluations.')

  flags.DEFINE_float('learning_rate', 1e-3, 'Initial learning rate.')

  flags.DEFINE_float('decay_factor', 0.33, 'Learning rate decay factor.')

  flags.DEFINE_float('decay_steps', 5000,
                     'Learning rate decay interval in steps.')

  flags.DEFINE_float('visit_weight', 1, 'Weight for visit loss.')
  flags.DEFINE_float('walker_weight', 1, 'Weight for walker loss.')
  flags.DEFINE_float('logit_weight', 1, 'Weight for logit loss.')
  flags.DEFINE_float('l1_weight', 0.001, 'Weight for embedding l1 regularization.')

  flags.DEFINE_integer('max_steps', 4000, 'Number of training steps.')
  flags.DEFINE_integer('warmup_steps', 3000, 'Number of warmup steps.')

  flags.DEFINE_integer('num_active_samples', 30, 'Number of active samples to request in total.')
  flags.DEFINE_integer('num_new_samples_per_run', 5, 'Number of active samples to request every run.')

  flags.DEFINE_string('logdir', '/tmp/semisup_mnist', 'Training log path.')
  flags.DEFINE_string('sample_method', 'propose_samples_pb_sampling', 'Method used to add new samples')

  flags.DEFINE_bool('semisup', True, 'Add unsupervised samples')

  flags.DEFINE_bool('restart', False, 'Restart from scratch after a new sample was added')

  flags.DEFINE_float('dropout_keep_prob', 1, 'Keep Prop in dropout. Set to 1 to deactivate dropout')

  print(FLAGS.learning_rate, FLAGS.__flags)  # print all flags (useful when logging)

from tools import mnist as mnist_tools
from backend import apply_envelope
import numpy as np

from tensorflow.python.framework import ops

NUM_LABELS = mnist_tools.NUM_LABELS
IMAGE_SHAPE = mnist_tools.IMAGE_SHAPE

def init_data():
  train_images, train_labels = mnist_tools.get_data('train')
  test_images, test_labels = mnist_tools.get_data('test')

  # Sample labeled training subset.
  seed = FLAGS.sup_seed if FLAGS.sup_seed != -1 else np.random.randint(0, 1000)
  print('Seed:', seed)

  sup_by_label = semisup.sample_by_label(train_images, train_labels,
                                         FLAGS.sup_per_class, NUM_LABELS, seed)

  return sup_by_label, train_images, train_labels, test_images, test_labels, seed


def init_graph(sup_by_label, train_images, train_labels, test_images, test_labels, logdir, unsup_batch_size=None):
  if unsup_batch_size == None:
    unsup_batch_size = FLAGS.unsup_batch_size

  graph = tf.Graph()
  with graph.as_default():
    model_func = semisup.architectures.mnist_model
    if FLAGS.dropout_keep_prob < 1:
      model_func = semisup.architectures.mnist_model_dropout

    model = semisup.SemisupModel(model_func, NUM_LABELS,
                                 IMAGE_SHAPE,
                                 dropout_keep_prob=FLAGS.dropout_keep_prob)

    # Set up inputs.
    t_sup_images = tf.placeholder("float", shape=[None] + IMAGE_SHAPE)
    t_sup_labels = tf.placeholder(dtype=tf.int32, shape=[None, ])

    # Compute embeddings and logits.
    t_sup_emb = model.image_to_embedding(t_sup_images)
    t_sup_logit = model.embedding_to_logit(t_sup_emb)

    t_unsup_images = tf.placeholder("float", shape=[None] + IMAGE_SHAPE)

    proximity_weight = tf.placeholder("float", shape=[])
    visit_weight = tf.placeholder("float", shape=[])
    walker_weight = tf.placeholder("float", shape=[])
    t_logit_weight = tf.placeholder("float", shape=[])
    t_l1_weight = tf.placeholder("float", shape=[])
    t_learning_rate = tf.placeholder("float", shape=[])

    t_unsup_emb = model.image_to_embedding(t_unsup_images)
    model.add_semisup_loss(
      t_sup_emb, t_unsup_emb, t_sup_labels,
      walker_weight=walker_weight, visit_weight=visit_weight, proximity_weight=proximity_weight, normalize_along_classes=True)
    model.add_logit_loss(t_sup_logit, t_sup_labels, weight=t_logit_weight)

    model.add_emb_regularization(t_sup_emb, weight=t_l1_weight)
    model.add_emb_regularization(t_unsup_emb, weight=t_l1_weight)

    train_op = model.create_train_op(t_learning_rate)
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(logdir, graph)

    saver = tf.train.Saver()
    unsup_images_iterator = semisup.create_input(train_images, train_labels,
                                                 unsup_batch_size)

    sess = tf.Session(graph=graph)
    sess.run(tf.global_variables_initializer())
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)

  def init_iterators():
    input_graph = tf.Graph()
    with input_graph.as_default():
      sup_images_iterator, sup_labels_iterator = semisup.create_per_class_inputs(
        sup_by_label, FLAGS.sup_per_batch)

      input_sess = tf.Session(graph=input_graph)
      input_sess.run(tf.global_variables_initializer())

    input_coord = tf.train.Coordinator()
    input_threads = tf.train.start_queue_runners(sess=input_sess, coord=input_coord)

    return sup_images_iterator, sup_labels_iterator, input_sess, input_threads, input_coord

  def train_warmup():
    """ if there are few samples, warmup at first"""
    sup_images_iterator, sup_labels_iterator, input_sess, input_threads, input_coord = init_iterators()

    model.reset_optimizer(sess)

    for step in range(FLAGS.max_steps):
      unsup_images, _ = sess.run(unsup_images_iterator)
      si, sl = input_sess.run([sup_images_iterator, sup_labels_iterator])

      _, summaries, train_loss = sess.run([train_op, summary_op, model.train_loss], {
        t_unsup_images: unsup_images,
        walker_weight: FLAGS.walker_weight,
        proximity_weight: 0,
        visit_weight: 0.1 + apply_envelope("lin", step, 0.4, FLAGS.warmup_steps, 0),
        t_logit_weight: 0.5,# FLAGS.logit_weight,
        t_l1_weight: FLAGS.l1_weight,
        t_sup_images: si,
        t_sup_labels: sl,
        t_learning_rate: 5e-5 + apply_envelope("log", step, FLAGS.learning_rate, FLAGS.warmup_steps, 0)
      })

      if (step + 1) % FLAGS.eval_interval == 0 or step == 99:
        print('Step: %d' % step)
        test_pred = model.classify(test_images, sess).argmax(-1)
        conf_mtx = semisup.confusion_matrix(test_labels, test_pred, NUM_LABELS)
        test_err = (test_labels != test_pred).mean() * 100
        print(conf_mtx)
        print('Test error: %.2f %%' % test_err)
        print('Train loss: %.2f ' % train_loss)
        print()

        test_summary = tf.Summary(
          value=[tf.Summary.Value(
            tag='Test Err', simple_value=test_err)])

        summary_writer.add_summary(summaries, step)
        summary_writer.add_summary(test_summary, step)

    input_coord.request_stop()
    input_coord.join(input_threads)
    input_sess.close()

  def train_finetune(lr=0.001, steps=FLAGS.max_steps):
    sup_images_iterator, sup_labels_iterator, input_sess, input_threads, input_coord = init_iterators()

    model.reset_optimizer(sess)
    for step in range(int(steps)):
      unsup_images, _ = sess.run(unsup_images_iterator)
      si, sl = input_sess.run([sup_images_iterator, sup_labels_iterator])
      _, summaries, train_loss = sess.run([train_op, summary_op, model.train_loss], {
        t_unsup_images: unsup_images,
        walker_weight: 1,
        proximity_weight: 0,
        visit_weight: 1,
        t_l1_weight: 0,
        t_logit_weight: 1,
        t_learning_rate: lr,
        t_sup_images: si,
        t_sup_labels: sl
      })
      if (step + 1) % FLAGS.eval_interval == 0 or step == 99:
        print('Step: %d' % step)
        test_pred = model.classify(test_images, sess).argmax(-1)
        conf_mtx = semisup.confusion_matrix(test_labels, test_pred, NUM_LABELS)
        test_err = (test_labels != test_pred).mean() * 100
        print(conf_mtx)
        print('Test error: %.2f %%' % test_err)
        print('Train loss: %.2f ' % train_loss)
        print()

        test_summary = tf.Summary(
          value=[tf.Summary.Value(
            tag='Test Err', simple_value=test_err)])

        summary_writer.add_summary(summaries, step)
        summary_writer.add_summary(test_summary, step)

    input_coord.request_stop()
    input_coord.join(input_threads)
    input_sess.close()

  def choose_sample(method="propose_samples"):
    sup_images_iterator, sup_labels_iterator, input_sess, input_threads, input_coord = init_iterators()

    sup_lbls = np.hstack([np.ones(len(i)) * ind for ind, i in enumerate(sup_by_label)])
    sup_images = np.vstack(sup_by_label)

    print("total number of supervised samples", sup_lbls.shape)

    n_samples = FLAGS.num_new_samples_per_run
    propose_func = getattr(model, method)
    inds_lba = propose_func(sup_images, sup_lbls, train_images, train_labels, sess, n_samples=n_samples)
    print('sampled new images with labels', train_labels[inds_lba])

    input_coord.request_stop()
    input_coord.join(input_threads)
    input_sess.close()

    return inds_lba

  def add_sample(index):
    # add sample
    label = train_labels[index]
    img = train_images[index]
    sup_by_label[label] = np.vstack([sup_by_label[label], [img]])

  return train_warmup, train_finetune, choose_sample, add_sample


def train_with_restart():
  indices = []
  choose_sample_iters = 1

  sup_by_label, train_images, train_labels, test_images, test_labels, seed = init_data()

  while True:
    logdir = FLAGS.logdir + '_' + str(seed) + '_' + str(len(indices))
    if len(indices) < 20: unsup_batch_size = FLAGS.unsup_batch_size
    else: unsup_batch_size = 100

    train_warmup, train_finetune, choose_sample, add_sample = \
      init_graph(sup_by_label, train_images, train_labels, test_images, test_labels, logdir, unsup_batch_size)

    if len(indices) < 20: train_warmup()
    else: train_finetune()

    for j in range(choose_sample_iters):
      inds = choose_sample(FLAGS.sample_method)

      chosen_inds = inds

      #print(chosen_inds)
      for ind in chosen_inds:
        indices.append(ind)
        add_sample(ind)

    if len(indices) >= FLAGS.num_active_samples:
      break


  # finalize
  train_warmup, train_finetune, choose_sample, add_sample = \
    init_graph(sup_by_label, train_images, train_labels, test_images, test_labels, logdir)

#  train_warmup()
  train_finetune()
  train_finetune(lr=0.0001)

  print('chosen labels', train_labels[indices])

  print('chosen indices', indices)

# only finetune
def train_without_restart():
  indices = []
  sup_by_label, train_images, train_labels, test_images, test_labels, seed = init_data()

  logdir = FLAGS.logdir + '_' + str(seed) + '_' + str(len(indices))
  train_warmup, train_finetune, choose_sample, add_sample = \
    init_graph(sup_by_label, train_images, train_labels, test_images, test_labels, logdir)
  train_warmup()

  for i in range(FLAGS.num_active_samples):
    ind = choose_sample()
    indices.append(ind)

    add_sample(ind)
    train_finetune(steps=FLAGS.max_steps / 2)
    train_finetune(lr=0.0003, steps=FLAGS.max_steps / 2)

  train_finetune(lr=0.0001)

  print('chosen indices', indices)

def main(_):
  if FLAGS.restart:
    train_with_restart()
  else:
    train_without_restart()

if __name__ == '__main__':
  app.run()
