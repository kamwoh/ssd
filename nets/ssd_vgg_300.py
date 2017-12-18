from collections import namedtuple

from tensorflow.contrib import slim

import tensorflow as tf

import numpy as np

import math

SSDParams = namedtuple('SSDParameters', ['img_shape',
                                         'num_classes',
                                         'no_annotation_label',
                                         'feat_layers',
                                         'feat_shapes',
                                         'anchor_size_bounds',
                                         'anchor_sizes',
                                         'anchor_ratios',
                                         'anchor_steps',
                                         'anchor_offset',
                                         'normalizations',
                                         'prior_scaling'
                                         ])


class SSDNet(object):
    def __init__(self):
        self.params = SSDParams(
                img_shape=(300, 300),
                num_classes=21,
                no_annotation_label=21,
                feat_layers=['conv4', 'conv7', 'conv8', 'conv9', 'conv10', 'conv11'],
                feat_shapes=[(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)],
                anchor_size_bounds=[0.15, 0.90],
                anchor_sizes=[(21., 45.),
                              (45., 99.),
                              (99., 153.),
                              (153., 207.),
                              (207., 261.),
                              (261., 315.)],
                anchor_ratios=[[2, .5],
                               [2, .5, 3, 1. / 3],
                               [2, .5, 3, 1. / 3],
                               [2, .5, 3, 1. / 3],
                               [2, .5],
                               [2, .5]],
                anchor_steps=[8, 16, 32, 64, 100, 300],
                anchor_offset=0.5,
                normalizations=[20, -1, -1, -1, -1, -1],
                prior_scaling=[0.1, 0.1, 0.2, 0.2]
        )

    def anchors(self):
        return ssd_anchors_all_layers(self.params.img_shape,
                                      self.params.feat_shapes,
                                      self.params.anchor_sizes,
                                      self.params.anchor_ratios,
                                      self.params.anchor_steps,
                                      self.params.anchor_offset)

    def arg_scope(self, weight_decay):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            biases_initializer=tf.zeros_initializer()):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                padding='SAME') as sc:
                return sc

    def bboxes_encode(self, labels, bboxes, anchors):
        with tf.name_scope('ssd_bboxes_encode'):
            self.target_labels = []
            self.target_localizations = []
            self.target_scores = []
            for i, anchors_layer in enumerate(anchors):
                with tf.name_scope('bboxes_encode_block_%i' % i):
                    t_labels, t_loc, t_scores = ssd_bboxes_encode_layer(labels, bboxes, anchors_layer,
                                                                        self.params.num_classes,
                                                                        self.params.no_annotation_label,
                                                                        self.params.prior_scaling)
                    self.target_labels.append(t_labels)
                    self.target_localizations.append(t_loc)
                    self.target_scores.append(t_scores)

            return self.target_labels, self.target_localizations, self.target_scores

    def losses(self, logits, localisations,
               gclasses, glocalisations, gscores,
               match_threshold,
               negative_ratio,
               loss_alpha,
               label_smoothing,
               device='/cpu:0'):
        with tf.name_scope('ssd_losses'):
            lshape = logits[0].get_shape().as_list()
            num_classes = lshape[-1]
            batch_size = lshape[0]

            flogits = []
            fgclasses = []
            fgscores = []
            flocalisations = []
            fglocalisations = []

            for i in range(len(logits)):
                flogits.append(tf.reshape(logits[i], [-1, num_classes]))
                fgclasses.append(tf.reshape(gclasses[i], [-1]))
                fgscores.append(tf.reshape(gscores[i], [-1]))
                flocalisations.append(tf.reshape(localisations[i], [-1, 4]))
                fglocalisations.append(tf.reshape(glocalisations[i], [-1, 4]))

            logits = tf.concat(flogits, axis=0)
            print logits
            gclasses = tf.concat(fgclasses, axis=0)
            print gclasses
            gscores = tf.concat(fgscores, axis=0)
            print gscores
            localisations = tf.concat(flocalisations, axis=0)
            print localisations
            glocalisations = tf.concat(fglocalisations, axis=0)
            print glocalisations
            dtype = logits.dtype

            # compute positive matching mask
            pmask = gscores > match_threshold
            print pmask
            fpmask = tf.cast(pmask, dtype)
            n_positives = tf.reduce_sum(fpmask)

            # hard negative mining
            no_classes = tf.cast(pmask, tf.int32)
            predictions = slim.softmax(logits)
            print predictions
            nmask = tf.logical_and(tf.logical_not(pmask),
                                   gscores > -0.5)
            print nmask
            fnmask = tf.cast(nmask, dtype)
            nvalues = tf.where(nmask,
                               predictions[:, 0],
                               1. - fnmask)
            print nvalues
            nvalues_flat = tf.reshape(nvalues, [-1])
            print nvalues_flat

            # number of negative entries to select
            max_neg_entries = tf.cast(tf.reduce_sum(fnmask), tf.int32)
            n_neg = tf.cast(negative_ratio * n_positives, tf.int32) + batch_size
            n_neg = tf.minimum(n_neg, max_neg_entries)
            print n_neg

            val, idxes = tf.nn.top_k(-nvalues_flat, k=n_neg)
            print val, idxes
            max_hard_pred = -val[-1]
            print max_hard_pred

            # final negative mask
            nmask = tf.logical_and(nmask, nvalues < max_hard_pred)
            fnmask = tf.cast(nmask, dtype)

            with tf.name_scope('cross_entropy_pos'):
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                      labels=gclasses)
                loss = tf.div(tf.reduce_sum(loss * fpmask), batch_size, name='value')
                tf.losses.add_loss(loss)

            with tf.name_scope('cross_entropy_neg'):
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                      labels=no_classes)
                loss = tf.div(tf.reduce_sum(loss * fnmask), batch_size, name='value')
                tf.losses.add_loss(loss)

            # add localisation loss: smooth l1
            with tf.name_scope('localisation'):
                weights = tf.expand_dims(loss_alpha * fpmask, axis=-1)
                x = localisations - glocalisations
                absx = tf.abs(x)
                minx = tf.minimum(absx, 1)
                loss = 0.5 * ((absx - 1) * minx + absx)
                loss = tf.div(tf.reduce_sum(loss * weights), batch_size, name='value')
                tf.losses.add_loss(loss)

    def ssd_net(self,
                inputs=None,
                dropout_keep_prob=0.5,
                is_training=True):
        if inputs is not None:
            self.inputs = inputs
        else:
            self.inputs = tf.placeholder(tf.float32,
                                         shape=[None,
                                                self.params.img_shape[0],
                                                self.params.img_shape[1],
                                                3],
                                         name='inputs')
        print self.inputs
        self.layers = {}
        with tf.variable_scope('ssd_300_vgg'):
            # vgg-16 blocks
            # block 1
            net = slim.repeat(self.inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            self.layers['conv1'] = net
            print net
            net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool1')
            self.layers['pool1'] = net
            print net

            # block 2
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            self.layers['conv2'] = net
            print net
            net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool2')
            self.layers['pool2'] = net
            print net

            # block 3
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            self.layers['conv3'] = net
            print net
            net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool3')
            self.layers['pool3'] = net
            print net

            # block 4
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            self.layers['conv4'] = net
            print net
            net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool4')
            self.layers['pool4'] = net
            print net

            # block 5
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            self.layers['conv5'] = net
            print net
            net = slim.max_pool2d(net, [3, 3], stride=1, padding='SAME', scope='pool5')
            self.layers['pool5'] = net
            print net

            # ssd blocks
            # block 6
            net = slim.conv2d(net, 1024, [3, 3], rate=6, scope='conv6')
            self.layers['conv6'] = net
            print net
            net = slim.dropout(net, keep_prob=dropout_keep_prob, is_training=is_training, scope='dropout6')
            self.layers['dropout6'] = net
            print net

            # block 7
            net = slim.conv2d(net, 1024, [1, 1], scope='conv7')
            self.layers['conv7'] = net
            print net
            net = slim.dropout(net, keep_prob=dropout_keep_prob, is_training=is_training, scope='dropout7')
            self.layers['dropout7'] = net
            print net

            with tf.variable_scope('block8'):
                net = slim.conv2d(net, 256, [1, 1], scope='conv8_1x1')
                print net
                net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], name='pad1x1')
                print net
                net = slim.conv2d(net, 512, [3, 3], stride=2, scope='conv8_3x3', padding='VALID')
                print net
            self.layers['conv8'] = net

            with tf.variable_scope('block9'):
                net = slim.conv2d(net, 128, [1, 1], scope='conv9_1x1')
                print net
                net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], name='pad9_1x1')
                print net
                net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv9_3x3', padding='VALID')
                print net
            self.layers['conv9'] = net

            with tf.variable_scope('block10'):
                net = slim.conv2d(net, 128, [1, 1], scope='conv10_1x1')
                print net
                net = slim.conv2d(net, 256, [3, 3], scope='conv10_3x3', padding='VALID')
                print net
            self.layers['conv10'] = net

            with tf.variable_scope('block11'):
                net = slim.conv2d(net, 128, [1, 1], scope='conv11_1x1')
                print net
                net = slim.conv2d(net, 256, [3, 3], scope='conv11_3x3', padding='VALID')
                print net
            self.layers['conv11'] = net

            self.predictions = []
            self.logits = []
            self.localisations = []
            for i, layer in enumerate(self.params.feat_layers):
                with tf.variable_scope(layer + '_box'):
                    p, l = ssd_multibox_layer(self.layers[layer],
                                              self.params.num_classes,
                                              self.params.anchor_sizes[i],
                                              self.params.anchor_ratios[i],
                                              self.params.normalizations[i])
                    print layer, 'class', p
                    print layer, 'location', l
                    self.predictions.append(slim.softmax(p))
                    self.logits.append(p)
                    self.localisations.append(l)

            return self.predictions, self.localisations, self.logits, self.layers


def ssd_bboxes_encode_layer(labels, bboxes, anchors_layer, num_classes, no_annotation_label,
                            prior_scaling, dtype=tf.float32):
    """
    Encode groundtruth labels and bounding boxes using SSD anchors from one layer.

    Arguments:
        labels: 1D Tensor(int64) containing groundtruth labels;
        bboxes: Nx4 Tensor(float) with bboxes relative coordinates;
        anchors_layer: Numpy array with layer anchors;
        matching_threshold: Threshold for positive match with groundtruth bboxes;
        prior_scaling: Scaling of encoded coordinates.

    Return:
        (target_labels, target_localizations, target_scores): Target Tensors.
    """
    # anchors coordinates and volume
    yref, xref, href, wref = anchors_layer
    ymin = yref - href / 2.
    xmin = xref - wref / 2.
    ymax = yref + href / 2.
    xmax = xref + wref / 2.
    vol_anchors = (xmax - xmin) * (ymax - ymin)

    shape = (yref.shape[0], yref.shape[1], href.size)  # grid_h, grid_w, 4 ?
    feature_labels = tf.zeros(shape, dtype=tf.int64)
    feature_scores = tf.zeros(shape, dtype=dtype)

    feature_ymin = tf.zeros(shape, dtype=dtype)
    feature_xmin = tf.zeros(shape, dtype=dtype)
    feature_ymax = tf.ones(shape, dtype=dtype)
    feature_xmax = tf.ones(shape, dtype=dtype)

    def jaccard_with_anchors(bbox):
        int_ymin = tf.maximum(ymin, bbox[0])
        int_xmin = tf.maximum(xmin, bbox[1])
        int_ymax = tf.minimum(ymax, bbox[2])
        int_xmax = tf.minimum(xmax, bbox[3])

        h = tf.maximum(int_ymax - int_ymin, 0.)
        w = tf.maximum(int_xmax - int_xmin, 0.)
        bbox_h = bbox[2] - bbox[0]
        bbox_w = bbox[3] - bbox[1]

        inter_vol = h * w
        union_vol = vol_anchors - inter_vol + bbox_h * bbox_w
        jaccard = tf.div(inter_vol, union_vol)
        return jaccard

    def condition(i, feat_labels, feat_scores,
                  feat_ymin, feat_xmin, feat_ymax, feat_xmax):
        # print 'labels shape ---', labels
        r = tf.less(i, tf.shape(labels))
        return r[0]

    def body(i, feat_labels, feat_scores,
             feat_ymin, feat_xmin, feat_ymax, feat_xmax):
        label = labels[i]
        bbox = bboxes[i]
        jaccard = jaccard_with_anchors(bbox)
        mask = tf.greater(jaccard, feat_scores)
        mask = tf.logical_and(mask, feat_scores > -0.5)
        mask = tf.logical_and(mask, label < num_classes)
        imask = tf.cast(mask, tf.int64)
        fmask = tf.cast(mask, dtype)

        feat_labels = imask * label + (1 - imask) * feat_labels
        feat_scores = tf.where(mask, jaccard, feat_scores)

        feat_ymin = fmask * bbox[0] + (1 - fmask) * feat_ymin
        feat_xmin = fmask * bbox[1] + (1 - fmask) * feat_xmin
        feat_ymax = fmask * bbox[2] + (1 - fmask) * feat_ymax
        feat_xmax = fmask * bbox[3] + (1 - fmask) * feat_xmax

        return [i + 1, feat_labels, feat_scores,
                feat_ymin, feat_xmin, feat_ymax, feat_xmax]

    i = 0
    [i, feature_labels, feature_scores,
     feature_ymin, feature_xmin,
     feature_ymax, feature_xmax] = tf.while_loop(condition, body, [i, feature_labels, feature_scores,
                                                                   feature_ymin, feature_xmin,
                                                                   feature_ymax, feature_xmax])

    # transform to center / size
    feature_cy = (feature_ymax + feature_ymin) / 2.
    feature_cx = (feature_xmax + feature_xmin) / 2.
    feature_h = feature_ymax - feature_ymin
    feature_w = feature_xmax - feature_xmin

    # encode features
    feature_cy = (feature_cy - yref) / href / prior_scaling[0]
    feature_cx = (feature_cx - xref) / wref / prior_scaling[1]
    feature_h = tf.log(feature_h / href) / prior_scaling[2]
    feature_w = tf.log(feature_w / wref) / prior_scaling[3]

    feature_localizations = tf.stack([feature_cx, feature_cy, feature_w, feature_h], axis=-1)

    # print 'feature labels', feature_labels
    # print 'feature localizations', feature_localizations
    # print 'feature scores', feature_scores

    return feature_labels, feature_localizations, feature_scores


def ssd_anchor_one_layer(img_shape,
                         feature_shape,
                         anchor_sizes,
                         anchor_ratios,
                         anchor_step,
                         offset=0.5,
                         dtype=np.float32):
    y, x = np.mgrid[0:feature_shape[0], 0:feature_shape[1]]
    y = (y.astype(dtype) + offset) * anchor_step / img_shape[0]
    x = (x.astype(dtype) + offset) * anchor_step / img_shape[1]

    y = np.expand_dims(y, axis=-1)
    x = np.expand_dims(x, axis=-1)

    num_anchors = len(anchor_sizes) + len(anchor_ratios)

    h = np.zeros((num_anchors,), dtype=dtype)
    w = np.zeros((num_anchors,), dtype=dtype)

    h[0] = anchor_sizes[0] / img_shape[0]
    w[0] = anchor_sizes[0] / img_shape[1]

    h[1] = math.sqrt(anchor_sizes[0] * anchor_sizes[1]) / img_shape[0]
    w[1] = math.sqrt(anchor_sizes[0] * anchor_sizes[1]) / img_shape[1]

    di = 2
    for i, r in enumerate(anchor_ratios):
        h[i + di] = anchor_sizes[0] / img_shape[0] / math.sqrt(r)
        w[i + di] = anchor_sizes[0] / img_shape[1] * math.sqrt(r)

    # y = center y of each grid
    # x = center x of each grid
    # h = height of each anchor
    # w = width of each anchor

    return y, x, h, w


def ssd_anchors_all_layers(img_shape,
                           layers_shape,
                           anchor_sizes,
                           anchor_ratios,
                           anchor_steps,
                           offset=0.5,
                           dtype=np.float32):
    layers_anchors = []
    for i, s in enumerate(layers_shape):
        anchor_bboxes = ssd_anchor_one_layer(img_shape,
                                             s,
                                             anchor_sizes[i],
                                             anchor_ratios[i],
                                             anchor_steps[i],
                                             offset,
                                             dtype)
        layers_anchors.append(anchor_bboxes)
    return layers_anchors


def ssd_multibox_layer(inputs,
                       num_classes,
                       anchor_sizes,
                       anchor_ratios,
                       anchor_normalization):
    # classifier net
    net = inputs
    if anchor_normalization > 0:
        net = l2_normalization(net, scaling=True)
        print net

    num_anchors = len(anchor_sizes) + len(anchor_ratios)

    num_loc_pred = num_anchors * 4
    # location
    loc_pred = slim.conv2d(net, num_loc_pred, [3, 3], activation_fn=None, scope='conv_loc')
    loc_pred = tf.reshape(loc_pred,
                          tensor_shape(loc_pred)[:-1] + [num_anchors, 4])  # 38x38x16 --> 38x38x4x4
    # class
    num_cls_pred = num_anchors * num_classes
    cls_pred = slim.conv2d(net, num_cls_pred, [3, 3], activation_fn=None, scope='conv_cls')
    cls_pred = tf.reshape(cls_pred, tensor_shape(cls_pred)[:-1] + [num_anchors, num_classes])  # 38x38x84 --> 38x38x4x21
    return cls_pred, loc_pred


def tensor_shape(x):
    if x.get_shape().is_fully_defined():
        return x.get_shape().as_list()
    else:
        static_shape = x.get_shape().with_rank(4).as_list()
        dynamic_shape = tf.unstack(tf.shape(x), 4)
        return [s if s is not None else d
                for s, d in zip(static_shape, dynamic_shape)]


def l2_normalization(inputs, scaling=False):
    with tf.variable_scope('l2_norm'):
        inputs_shape = inputs.get_shape()
        inputs_rank = inputs_shape.ndims
        dtype = inputs.dtype.base_dtype
        norm_dim = tf.range(inputs_rank - 1, inputs_rank)
        params_shape = inputs_shape[-1:]
        outputs = tf.nn.l2_normalize(inputs, norm_dim)
        if scaling:
            scale = tf.get_variable('gamma',
                                    shape=params_shape,
                                    dtype=dtype,
                                    initializer=tf.ones_initializer())
            outputs = tf.multiply(outputs, scale)
            return outputs
