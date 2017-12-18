import math

smin = 0.10
smax = 0.9

m = 6

for k in xrange(1, m + 1):
    res = smax - smin
    res /= (m - 1)
    res *= (k - 1)
    print smin + res

min_ratio = 20
max_ratio = 90
step = int(math.floor((max_ratio - min_ratio) / (6 - 2)))
# print step
print
min_dim = 300
min_sizes = []
max_sizes = []
for ratio in xrange(min_ratio, max_ratio + 1, step):
    print ratio
    print ratio / 100.
    print (ratio + step) / 100.
    print
    min_sizes.append(min_dim * ratio / 100.)
    max_sizes.append(min_dim * (ratio + step) / 100.)

min_sizes = [min_dim * (min_ratio / 2) / 100.] + min_sizes
max_sizes = [min_dim * min_ratio / 100.] + max_sizes

print min_sizes
print max_sizes

import tensorflow as tf


def body(x, y):
    return x + 2, [y[0] + 1, y[1] + 10]


def condition(x, y):
    r = tf.less(x, y)
    print r[0]
    return r[0]


import cv2
import numpy as np

sample_img = 'datasets/VOCdevkit_train/VOC2007/JPEGImages/000005.jpg'
img = cv2.imread(sample_img)

bboxes = np.array([[211., 263., 339., 324.], [264., 165., 372., 253.]])

shape = [375, 500, 3]

converted_bboxes = []

for y1, x1, y2, x2 in bboxes:
    y1 /= shape[0]
    y2 /= shape[0]
    x1 /= shape[1]
    x2 /= shape[1]
    converted_bboxes.append((y1, x1, y2, x2))
print converted_bboxes
tfimg = tf.constant(img)
tfshape = tf.constant(shape)
print tfshape
tfbboxes = tf.constant(converted_bboxes, tf.float32)

bbox_begin, bbox_size, distort_bbox = tf.image.sample_distorted_bounding_box(tfshape,
                                                                             bounding_boxes=tf.expand_dims(tfbboxes, 0),
                                                                             min_object_covered=0.25,
                                                                             area_range=(0.1, 1.0),
                                                                             max_attempts=200,
                                                                             use_image_if_no_bounding_boxes=True)
image_with_bbox = tf.image.draw_bounding_boxes(tf.expand_dims(tf.cast(tfimg, tf.float32), 0), distort_bbox)
print distort_bbox
cropped_image = tf.slice(tfimg, bbox_begin, bbox_size)
print cropped_image
cropped_image.set_shape([None, None, 3])
print cropped_image

with tf.Session() as sess:
    # print sess.run(bbox_begin)
    # print sess.run(bbox_size)
    # print sess.run(distort_bbox)
    # print sess.run(distort_bbox)
    sessimg, sessbb, sessimg_bb = sess.run([cropped_image, distort_bbox, image_with_bbox])
    print sessbb, sessimg_bb
    cropped_shape = sessimg.shape
    # cropped_shape = sess.run(cropped_image).shape
    print cropped_shape
    y1, x1, y2, x2 = sessbb[0, 0]
    print y1, x1, y2, x2
    y1 *= cropped_shape[0]
    y2 *= cropped_shape[0]
    x1 *= cropped_shape[1]
    x2 *= cropped_shape[1]

    print y1, x1, y2, x2
    sessimg = cv2.resize(sessimg, (shape[1], shape[0]))
    cv2.rectangle(sessimg, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    for bbox in bboxes:
        cv2.rectangle(img, tuple([int(v) for v in bbox[0:2]][::-1]), tuple([int(v) for v in bbox[2:4]][::-1]),
                      (0, 255, 0), 2)
    cv2.imshow('ori', img)
    cv2.imshow('test', sessimg)
    # sessimg_bb = sessimg_bb.astype(np.uint8)[0]
    # cv2.imshow('shit', sessimg_bb)
    cv2.waitKey(0)
import os

print os.path.abspath(__file__)
