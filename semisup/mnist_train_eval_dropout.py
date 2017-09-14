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

Association-based semi-supervised training example in MNIST dataset.

Training should reach ~1% error rate on the test set using 100 labeled samples
in 5000-10000 steps (a few minutes on Titan X GPU)

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

flags.DEFINE_integer('sup_per_class', 3,
                     'Number of labeled samples used per class.')

flags.DEFINE_integer('sup_seed', -1,
                     'Integer random seed used for labeled set selection.')

flags.DEFINE_integer('sup_per_batch', 3,
                     'Number of labeled samples per class per batch.')

flags.DEFINE_integer('unsup_batch_size', 50,
                     'Number of unlabeled samples per batch.')

flags.DEFINE_integer('eval_interval', 500,
                     'Number of steps between evaluations.')

flags.DEFINE_float('learning_rate', 1e-3, 'Initial learning rate.')

flags.DEFINE_float('decay_factor', 0.33, 'Learning rate decay factor.')

flags.DEFINE_float('decay_steps', 20000,
                   'Learning rate decay interval in steps.')

flags.DEFINE_float('visit_weight', 1, 'Weight for visit loss.')
flags.DEFINE_float('walker_weight', 1, 'Weight for walker loss.')
flags.DEFINE_float('logit_weight', 0.5, 'Weight for logit loss.')
flags.DEFINE_float('l1_weight', 0.001, 'Weight for embedding l1 regularization.')

flags.DEFINE_integer('max_steps', 7000, 'Number of training steps.')
flags.DEFINE_integer('warmup_steps', 3000, 'Number of warmup steps.')

flags.DEFINE_string('logdir', '/tmp/semisup_mnist', 'Training log path.')

flags.DEFINE_bool('semisup', True, 'Add unsupervised samples')

flags.DEFINE_float('dropout_keep_prob', 0.8, 'Keep Prop in dropout. Set to 1 to deactivate dropout')

print(FLAGS.learning_rate, FLAGS.__flags) # print all flags (useful when logging)

from tools import mnist as mnist_tools
from backend import apply_envelope
import numpy as np

NUM_LABELS = mnist_tools.NUM_LABELS
IMAGE_SHAPE = mnist_tools.IMAGE_SHAPE


def main(_):
  FLAGS.emb_size = 128
  optimizer = 'adam'

  train_images, train_labels = mnist_tools.get_data('train')
  test_images, test_labels = mnist_tools.get_data('test')

  # Sample labeled training subset.
  seed = FLAGS.sup_seed if FLAGS.sup_seed != -1 else np.random.randint(0, 1000)
  print('Seed:', seed)

  sup_by_label = semisup.sample_by_label(train_images, train_labels,
                                         FLAGS.sup_per_class, NUM_LABELS, seed)

  graph = tf.Graph()
  with graph.as_default():
    model_func = semisup.architectures.mnist_model
    if FLAGS.dropout_keep_prob < 1:
        model_func = semisup.architectures.mnist_model_dropout

    model = semisup.SemisupModel(model_func, NUM_LABELS,
                                 IMAGE_SHAPE, optimizer=optimizer, emb_size=FLAGS.emb_size,
                                 dropout_keep_prob=FLAGS.dropout_keep_prob)

    # Set up inputs.
    t_sup_images, t_sup_labels = semisup.create_per_class_inputs(
      sup_by_label, FLAGS.sup_per_batch)

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
      walker_weight=walker_weight, visit_weight=visit_weight, proximity_weight=proximity_weight)
    model.add_logit_loss(t_sup_logit, t_sup_labels, weight=t_logit_weight)

    model.add_emb_regularization(t_sup_emb, weight=t_l1_weight)
    model.add_emb_regularization(t_unsup_emb, weight=t_l1_weight)

    train_op = model.create_train_op(t_learning_rate)
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(FLAGS.logdir, graph)

    saver = tf.train.Saver()

  sess = tf.InteractiveSession(graph=graph)

  unsup_images_iterator = semisup.create_input(train_images, train_labels,
                                               FLAGS.unsup_batch_size)
  tf.global_variables_initializer().run()

  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)

  use_new_visit_loss = False
  for step in range(FLAGS.max_steps):
    unsup_images, _ = sess.run(unsup_images_iterator)

    if use_new_visit_loss:
      _, summaries, train_loss = sess.run([train_op, summary_op, model.train_loss], {
        t_unsup_images: unsup_images,
        walker_weight: FLAGS.walker_weight,
        proximity_weight: 0.3 + apply_envelope("lin", step, 0.7, FLAGS.warmup_steps, 0)
                          - apply_envelope("lin", step, FLAGS.visit_weight, 2000, FLAGS.warmup_steps),
        t_logit_weight: FLAGS.logit_weight,
        t_l1_weight: FLAGS.l1_weight,
        visit_weight: apply_envelope("lin", step, FLAGS.visit_weight, 2000, FLAGS.warmup_steps),
        t_learning_rate: 5e-5 + apply_envelope("log", step, FLAGS.learning_rate, FLAGS.warmup_steps, 0)
      })
    else:
      _, summaries, train_loss = sess.run([train_op, summary_op, model.train_loss], {
        t_unsup_images: unsup_images,
        walker_weight: FLAGS.walker_weight,
        visit_weight: 0.3 + apply_envelope("lin", step, 0.7, FLAGS.warmup_steps, 0),
        proximity_weight: 0,
        t_logit_weight: FLAGS.logit_weight,
        t_l1_weight: FLAGS.l1_weight,
        t_learning_rate: 5e-5 + apply_envelope("log", step, FLAGS.learning_rate, FLAGS.warmup_steps, 0)
      })

    if (step + 1) % FLAGS.eval_interval == 0 or step == 99:
      print('Step: %d' % step)
      test_pred = model.classify(test_images).argmax(-1)
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


if __name__ == '__main__':
  app.run()
