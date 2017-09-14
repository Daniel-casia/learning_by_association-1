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

import os, math
import tensorflow as tf
import semisup

from tensorflow.python.platform import app
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

flags.DEFINE_integer('sup_per_class', 50,
                     'Number of labeled samples used per class.')

flags.DEFINE_integer('sup_seed', 47,
                     'Integer random seed used for labeled set selection.')

flags.DEFINE_integer('sup_per_batch', 1,
                     'Number of labeled samples per class per batch.')

flags.DEFINE_integer('unsup_batch_size', 100,
                     'Number of unlabeled samples per batch.')

flags.DEFINE_integer('eval_interval', 200,
                     'Number of epochs between evaluations.')


flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')

flags.DEFINE_float('decay_factor', 0.1, 'Learning rate decay factor.')

flags.DEFINE_float('decay_epochs', 150,
                   'Learning rate decay interval in epochs.')

flags.DEFINE_float('decay_steps', 5000,
                   'Learning rate decay interval in steps.')

flags.DEFINE_float('visit_weight', 1.0, 'Weight for visit loss.')
flags.DEFINE_float('proximity_weight', 0, 'Weight for proximity loss.')

flags.DEFINE_float('walker_weight', 1.0, 'Weight for walker loss.')

flags.DEFINE_integer('max_epochs', 300, 'Number of training epochs.')
flags.DEFINE_integer('max_steps', 20000, 'Number of training steps.')

flags.DEFINE_string('logdir', '/tmp/semisup_cifar', 'Training log path.')
flags.DEFINE_string('model', 'mnist_model', 'Which architecture to use')

flags.DEFINE_string('cifar', 'cifar100', 'Which cifar dataset to use')
flags.DEFINE_bool('equal_cls_unsup', False, 'Enforce an equal distribution of unsup samples')
flags.DEFINE_bool('semisup', True, 'Add unsupervised samples')
flags.DEFINE_bool('random_batches', False, 'Sample examples randomly instead of fixed per class batch size')
flags.DEFINE_bool('proximity_loss', False, 'Use proximity loss instead of visit loss')

flags.DEFINE_float('dropout_keep_prob', 1, 'Keep Prop in dropout. Set to 1 to deactivate dropout')

from tools import cifar_inmemory as data
from backend import apply_envelope

print(FLAGS.learning_rate, FLAGS.__flags) # print all flags (useful when logging)

IMAGE_SHAPE = [32, 32, 3]
NUM_TRAIN_IMAGES = 50000

if FLAGS.cifar == 'cifar10':
  NUM_LABELS = 10
elif FLAGS.cifar == 'cifar100':
  NUM_LABELS = 100
elif FLAGS.cifar == 'cifar100coarse':
  NUM_LABELS = 20
else:
  print('dataset not supported')

import numpy as np

def main(_):

# Sample labeled training subset.

  train_images, train_labels = data.load_training_data(FLAGS.cifar)
  test_images, test_labels = data.load_test_data(FLAGS.cifar)

  print(train_images.shape, train_labels.shape)

  # Sample labeled training subset.
  seed = FLAGS.sup_seed if FLAGS.sup_seed != -1 else np.random.randint(0, 1000)
  print('Seed:', seed)
  sup_by_label = semisup.sample_by_label(train_images, train_labels,
                                         FLAGS.sup_per_class, NUM_LABELS, seed)

  graph = tf.Graph()
  with graph.as_default():

    def augment(image):
        # image_size = 28
        # image = tf.image.resize_image_with_crop_or_pad(
        #    image, image_size+4, image_size+4)
        # image = tf.random_crop(image, [image_size, image_size, 3])
        image = tf.image.random_flip_left_right(image)
        # Brightness/saturation/constrast provides small gains .2%~.5% on cifar.
        # image = tf.image.random_brightness(image, max_delta=63. / 255.)
        # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        # image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
        return image

    model_func = getattr(semisup.architectures, FLAGS.model)
    model = semisup.SemisupModel(model_func, NUM_LABELS,
                                 IMAGE_SHAPE, emb_size=256, dropout_keep_prob=FLAGS.dropout_keep_prob,
                                 augmentation_function=augment)


    if FLAGS.random_batches:
      sup_lbls = np.asarray(np.hstack([np.ones(len(i)) * ind for ind, i in enumerate(sup_by_label)]), np.int)
      sup_images = np.vstack(sup_by_label)
      batch_size = FLAGS.sup_per_batch * NUM_LABELS
      t_sup_images, t_sup_labels = semisup.create_input(sup_images, sup_lbls, batch_size)
    else:
      t_sup_images, t_sup_labels = semisup.create_per_class_inputs(
        sup_by_label, FLAGS.sup_per_batch)

    # Compute embeddings and logits.
    t_sup_emb = model.image_to_embedding(t_sup_images)
    t_sup_logit = model.embedding_to_logit(t_sup_emb)

    # Add losses.
    if FLAGS.semisup:
      if FLAGS.equal_cls_unsup:
        allimgs_bylabel = semisup.sample_by_label(train_images, train_labels,
                                                  int(NUM_TRAIN_IMAGES / NUM_LABELS), NUM_LABELS, seed)
        t_unsup_images, _ = semisup.create_per_class_inputs(
          allimgs_bylabel, FLAGS.sup_per_batch)
      else:
        t_unsup_images, _ = semisup.create_input(train_images, train_labels,
                                                 FLAGS.unsup_batch_size)

      t_unsup_emb = model.image_to_embedding(t_unsup_images)
      proximity_weight = tf.placeholder("float", shape=[])
      visit_weight = tf.placeholder("float", shape=[])
      walker_weight = tf.placeholder("float", shape=[])

      model.add_semisup_loss(
        t_sup_emb, t_unsup_emb, t_sup_labels,
        walker_weight=walker_weight, visit_weight=visit_weight, proximity_weight=proximity_weight)
    model.add_logit_loss(t_sup_logit, t_sup_labels)

    t_learning_rate = tf.train.exponential_decay(
      FLAGS.learning_rate,
      model.step,
      FLAGS.decay_steps,
      FLAGS.decay_factor,
      staircase=True)
    train_op = model.create_train_op(t_learning_rate)
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(FLAGS.logdir, graph)

    saver = tf.train.Saver()
  with tf.Session(graph=graph) as sess:
    tf.global_variables_initializer().run()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for step in range(FLAGS.max_steps):
      feed_dict={}
      if FLAGS.semisup:

        if FLAGS.proximity_loss:
          feed_dict = {
            walker_weight:  FLAGS.walker_weight,#0.1 + apply_envelope("lin", step, FLAGS.walker_weight, 500, FLAGS.decay_steps),
            visit_weight: 0,
            proximity_weight: FLAGS.visit_weight#0.1 + apply_envelope("lin", step,FLAGS.visit_weight, 500, FLAGS.decay_steps),
          #t_logit_weight: FLAGS.logit_weight,
          #t_learning_rate: 5e-5 + apply_envelope("log", step, FLAGS.learning_rate, FLAGS.warmup_steps, 0)
          }
        else:
          feed_dict =  {
            walker_weight:  FLAGS.walker_weight,#0.1 + apply_envelope("lin", step, FLAGS.walker_weight, 500, FLAGS.decay_steps),
            visit_weight: FLAGS.visit_weight, #0.1 + apply_envelope("lin", step,FLAGS.visit_weight, 500, FLAGS.decay_steps),
            proximity_weight: 0
          #t_logit_weight: FLAGS.logit_weight,
          #t_learning_rate: 5e-5 + apply_envelope("log", step, FLAGS.learning_rate, FLAGS.warmup_steps, 0)
          }

      _, summaries, train_loss = sess.run([train_op, summary_op, model.train_loss], feed_dict)


      if (step + 1) % FLAGS.eval_interval == 0 or step == 99:
        print('Step: %d' % step)
        test_pred = model.classify(test_images, sess).argmax(-1)
        conf_mtx = semisup.confusion_matrix(test_labels, test_pred, NUM_LABELS)
        test_err = (test_labels != test_pred).mean() * 100
        print(conf_mtx)
        print('Test error: %.2f %%' % test_err)
        print()

        test_summary = tf.Summary(
            value=[tf.Summary.Value(
                tag='Test Err', simple_value=test_err)])

        summary_writer.add_summary(summaries, step)
        summary_writer.add_summary(test_summary, step)

        #saver.save(sess, FLAGS.logdir, model.step)

    coord.request_stop()
    coord.join(threads)


if __name__ == '__main__':
  app.run()
