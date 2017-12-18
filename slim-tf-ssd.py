from collections import namedtuple
import numpy as np
import math
from nets import ssd_vgg_300
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

from tensorflow.contrib import slim
from datasets import dataset as voc_dataset
from preprocessing import ssd_vgg_preprocessing
import os

LearningRateParams = namedtuple('LearningRateParams', ['learning_rate',
                                                       'end_learning_rate',
                                                       'learning_rate_decay_factor',
                                                       'num_epochs_per_decay'])

SSDParams = namedtuple('SSDParams', ['loss_alpha',
                                     'negative_ratio',
                                     'match_threshold',
                                     'label_smoothing'])

TrainParams = namedtuple('TrainParams', ['weight_decay',
                                         'batch_size',
                                         'optimizer',
                                         'adam_beta1',
                                         'adam_beta2',
                                         'opt_epsilon'])


def reshape_list(l, shape=None):
    r = []
    if shape is None:
        for a in l:
            if isinstance(a, (list, tuple)):
                r = r + list(a)
            else:
                r.append(a)
    else:
        i = 0
        for s in shape:
            if s == 1:
                r.append(l[i])
            else:
                r.append(l[i:i + s])
            i += s
    return r


def main_train():
    dirname = os.path.dirname(__file__)
    dataset_dir = '{}/datasets/VOC_SSD'.format(dirname)
    ssd_net = ssd_vgg_300.SSDNet()

    train_params = TrainParams(weight_decay=0.00004,
                               batch_size=32,
                               optimizer='adam',
                               adam_beta1=0.9,
                               adam_beta2=0.999,
                               opt_epsilon=1.0)

    ssd_params = SSDParams(loss_alpha=1.,
                           negative_ratio=3.,
                           match_threshold=0.5,
                           label_smoothing=0.0)

    learning_rate_params = LearningRateParams(learning_rate=0.01,
                                              end_learning_rate=0.0001,
                                              learning_rate_decay_factor=0.94,
                                              num_epochs_per_decay=2.0)

    graph = tf.Graph()
    with graph.as_default():
        with tf.device('/device:CPU:0'):
            global_step = tf.train.create_global_step()

        dataset = voc_dataset.get_dataset(dataset_dir, 'train')
        ssd_shape = ssd_net.params.img_shape

        print '---- enter anchors ----'

        ssd_anchors = ssd_net.anchors()

        print '---- end anchors ----'

        image_preprocessing_fn = ssd_vgg_preprocessing.preprocess_for_train

        provider = voc_dataset.get_dataset_provider(dataset, 4, train_params.batch_size)

        [image, shape, glabels, gbboxes] = provider.get(['image', 'shape',
                                                         'object/label',
                                                         'object/bbox'])

        print '---- from provider ----'
        print image
        print shape
        print glabels
        print gbboxes
        print '---- end provider ----'

        print '---- enter preprocessing ----'
        image, glabels, gbboxes = image_preprocessing_fn(image, glabels, gbboxes, ssd_shape, 'NHWC')
        print '---- end preprocessing ----'

        print '---- encode bbox ----'
        gclasses, glocalisations, gscores = ssd_net.bboxes_encode(glabels, gbboxes, ssd_anchors)
        print '---- end encode ----'

        batch_shape = [1] + [len(ssd_anchors)] * 3

        print batch_shape

        tensors = reshape_list([image, gclasses, glocalisations, gscores])

        print 'reshape 1 ---', tensors, len(tensors)

        r = tf.train.batch(tensors, train_params.batch_size, num_threads=4, capacity=5 * train_params.batch_size)

        print 'train batch ---', r, len(r)

        b_image, b_gclasses, b_glocalisations, b_gscores = reshape_list(r, batch_shape)

        print b_image
        print b_gclasses
        print b_glocalisations
        print b_gscores

        tensors = reshape_list([b_image, b_gclasses, b_glocalisations, b_gscores])

        print 'reshape 2 ---', tensors, len(tensors)

        batch_queue = slim.prefetch_queue.prefetch_queue(tensors, capacity=2)

        print 'batch queue ---', batch_queue

        b_image, b_gclasses, b_glocalisations, b_gscores = reshape_list(batch_queue.dequeue(), batch_shape)
        arg_scope = ssd_net.arg_scope(weight_decay=train_params.weight_decay)

        with slim.arg_scope(arg_scope):
            predictions, localisations, logits, end_points = ssd_net.ssd_net(b_image, is_training=True)

            ssd_net.losses(logits, localisations, b_gclasses, b_glocalisations, b_gscores,
                           match_threshold=ssd_params.match_threshold,
                           negative_ratio=ssd_params.negative_ratio,
                           loss_alpha=ssd_params.loss_alpha,
                           label_smoothing=ssd_params.label_smoothing)

        # moving_average_variables = slim.get_model_variables()
        # variable_averages = tf.train.ExponentialMovingAverage(0.999, global_step)
        decay_steps = int(dataset.num_samples / train_params.batch_size * learning_rate_params.num_epochs_per_decay)

        learning_rate = tf.train.exponential_decay(learning_rate_params.learning_rate,
                                                   global_step,
                                                   decay_steps,
                                                   learning_rate_params.learning_rate_decay_factor,
                                                   staircase=True)

        optimizer = tf.train.AdamOptimizer(learning_rate,
                                           train_params.adam_beta1,
                                           train_params.adam_beta2,
                                           train_params.opt_epsilon)

        print '---- loss ----'
        losses = tf.get_collection(tf.GraphKeys.LOSSES)
        print losses
        loss = tf.add_n(losses, name='loss')
        print loss
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        print regularization_losses
        regularization_loss = tf.add_n(regularization_losses, name='regularization_loss')
        print regularization_loss
        sum_loss = tf.add_n([loss, regularization_loss])
        print sum_loss
        print '---- end ----'

        grad = optimizer.compute_gradients(sum_loss)

        grad_updates = optimizer.apply_gradients(grad, global_step)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        print 'update ops', update_ops

        update_ops.append(grad_updates)

        update_op = tf.group(*update_ops)

        train_tensor = control_flow_ops.with_dependencies([update_op], sum_loss, name='train_op')

        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(gpu_options=gpu_options)
        saver = tf.train.Saver(max_to_keep=10,
                               keep_checkpoint_every_n_hours=1.0)

        import sys
        # with tf.Session(config=config) as sess:
            # epochs = 10
            # sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            # for epoch in xrange(epochs):
                # print '%s/%s' % (epoch + 1, epochs)
                # tf.train.start_queue_runners(sess=sess)
                # steps = int(dataset.num_samples / train_params.batch_size)
                # for step in xrange(steps):
                    # sys.stdout.write('\r%s/%s' % (step + 1, steps))
                    # sys.stdout.flush()
                    # _, loss = sess.run([train_tensor, sum_loss])
                # print
        slim.learning.train(train_tensor,
                            '{}/trained_model'.format(dirname),
                            is_chief=True,
                            init_fn=None,
                            saver=saver,
                            session_config=config)


if __name__ == '__main__':
    main_train()
