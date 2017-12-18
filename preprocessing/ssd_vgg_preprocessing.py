import tensorflow as tf
import tf_ip
from tensorflow.python.ops import control_flow_ops

# VGG mean parameters.
_R_MEAN = 123.
_G_MEAN = 117.
_B_MEAN = 104.

MIN_OBJECT_COVERED = 0.25
CROP_RATIO_RANGE = (0.6, 1.67)  # Distortion ratio during cropping.
EVAL_SIZE = (300, 300)


def apply_with_random_selector(x, func, num_cases):
    sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
    return control_flow_ops.merge([func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
                                      for case in range(num_cases)])[0]


def preprocess_for_train(image, labels, bboxes, out_shape, data_format):
    fast_mode = False
    with tf.name_scope('ssd_preprocessing_train'):
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        # tf_summary_image(image, bboxes, 'image_with_bboxes')

        dst_image = image

        # distort image and bounding boxes
        dst_image, labels, bboxes, distort_bbox = tf_ip.distorted_bounding_box_crop(image, labels, bboxes,
                                                                                    min_object_covered=MIN_OBJECT_COVERED,
                                                                                    aspect_ratio_range=CROP_RATIO_RANGE)

        # resize
        dst_image = tf_ip.resize_image(dst_image, out_shape)

        # tf_summary_image(dst_image, bboxes, 'image_shape_distorted')

        # random flip
        dst_image, bboxes = tf_ip.random_flip_left_right(dst_image, bboxes)

        # distort colors
        dst_image = apply_with_random_selector(dst_image,
                                               lambda x, ordering: tf_ip.distort_color(x, ordering, fast_mode),
                                               num_cases=4)

        # tf_summary_image(dst_image, bboxes, 'image_color_distorted')

        image = dst_image * 255
        image = tf_ip.image_whitened(image, [_R_MEAN, _G_MEAN, _B_MEAN])

    return image, labels, bboxes
