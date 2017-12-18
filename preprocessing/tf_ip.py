import tensorflow as tf
import numpy as np

BBOX_CROP_OVERLAP = 0.5  # Minimum overlap to keep a bbox after cropping.


def resize_image(image, size, method=tf.image.ResizeMethod.BILINEAR, align_corners=False):
    with tf.name_scope('resize_image'):
        channels = image.get_shape().as_list()[2]
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_images(image, size, method, align_corners)
        image = tf.reshape(image, tf.stack([size[0], size[1], channels]))
        return image


def random_flip_left_right(image, bboxes, seed=np.random.randint(100000)):
    def flip_bboxes(bboxes):
        bboxes = tf.stack([bboxes[:, 0], 1 - bboxes[:, 3], bboxes[:, 2], 1 - bboxes[:, 1]],
                          axis=-1)  # flip xmin and xmax
        return bboxes

    with tf.name_scope('random_flip_left_right'):
        image = tf.convert_to_tensor(image, name='image')
        uniform_random = tf.random_uniform([], 0.0, 1.0, seed=seed)
        mirror_cond = tf.less(uniform_random, 0.5)

        flipped_image = tf.cond(mirror_cond,
                                lambda: tf.reverse_v2(image, [1]),
                                lambda: image)  # flip image

        bboxes = tf.cond(mirror_cond,
                         lambda: flip_bboxes(bboxes),
                         lambda: bboxes)

        flipped_image.set_shape(image.get_shape())

        return flipped_image, bboxes


def distort_color(image, color_ordering=0, fast_mode=True, scope=None):
    """Distort the color of a Tensor image.

    Each color distortion is non-commutative and thus ordering of the color ops
    matters. Ideally we would randomly permute the ordering of the color ops.
    Rather then adding that level of complication, we select a distinct ordering
    of color ops for each preprocessing thread.

    Args:
        image: 3-D Tensor containing single image in [0, 1].
        color_ordering: Python int, a type of distortion (valid values: 0-3).
        fast_mode: Avoids slower ops (random_hue and random_contrast)
        scope: Optional scope for name_scope.
    Returns:
        3-D Tensor color-distorted image on range [0, 1]
    Raises:
        ValueError: if color_ordering not in [0, 3]
    """
    with tf.name_scope(scope, 'distort_color', [image]):
        if fast_mode:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            else:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
        else:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            elif color_ordering == 1:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
            elif color_ordering == 2:
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            elif color_ordering == 3:
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
            else:
                raise ValueError('color_ordering must be in [0, 3]')
        # The random_* ops do not necessarily clamp.
        return tf.clip_by_value(image, 0.0, 1.0)


def safe_divide(numerator, denominator, name):
    """
    Divides two values, returning 0 if the denominator is <= 0.
    Args:
        numerator: A real `Tensor`.
        denominator: A real `Tensor`, with dtype matching `numerator`.
        name: Name for the returned op.
    Returns:
        0 if `denominator` <= 0, else `numerator` / `denominator`
    """
    return tf.where(tf.greater(denominator, 0),
                    tf.divide(numerator, denominator),
                    tf.zeros_like(numerator),
                    name=name)


def bboxes_intersection(bbox_ref, bboxes):
    with tf.name_scope('bboxes_intersection'):
        bboxes = tf.transpose(bboxes)
        bbox_ref = tf.transpose(bbox_ref)

        # intersection bbox and volume
        int_ymin = tf.maximum(bboxes[0], bbox_ref[0])
        int_xmin = tf.maximum(bboxes[1], bbox_ref[1])
        int_ymax = tf.minimum(bboxes[2], bbox_ref[2])
        int_xmax = tf.minimum(bboxes[3], bbox_ref[3])

        int_h = tf.maximum(int_ymax - int_ymin, 0.)
        int_w = tf.maximum(int_xmax - int_xmin, 0.)
        bbox_w = bboxes[2] - bboxes[0]
        bbox_h = bboxes[3] - bboxes[1]

        inter_vol = int_h * int_w
        bboxes_vol = bbox_h * bbox_w
        scores = safe_divide(inter_vol, bboxes_vol, 'intersection')
        return scores


def bboxes_filter_overlap(labels, bboxes, threshold):
    """
    Filter out bounding boxes based on (relative )overlap with reference
    box [0, 0, 1, 1].  Remove completely bounding boxes, or assign negative
    labels to the one outside (useful for latter processing...).

    Return:
        labels, bboxes: Filtered (or newly assigned) elements.
    """
    with tf.name_scope('bboxes_filter'):
        scores = bboxes_intersection(tf.constant([0, 0, 1, 1], bboxes.dtype), bboxes)
        mask = scores > threshold

        labels = tf.boolean_mask(labels, mask)
        bboxes = tf.boolean_mask(bboxes, mask)
        return labels, bboxes


def bboxes_resize(bbox_ref, bboxes):
    """Resize bounding boxes based on a reference bounding box,
        assuming that the latter is [0, 0, 1, 1] after transform. Useful for
        updating a collection of boxes after cropping an image.
        """
    with tf.name_scope('bboxes_resize'):
        # translate
        v = tf.stack([bbox_ref[0], bbox_ref[1], bbox_ref[2], bbox_ref[3]])
        bboxes = bboxes - v

        # scale
        s = tf.stack([bbox_ref[2] - bbox_ref[0],
                      bbox_ref[3] - bbox_ref[1],
                      bbox_ref[2] - bbox_ref[0],
                      bbox_ref[3] - bbox_ref[1]])
        bboxes = bboxes / s
        return bboxes


def distorted_bounding_box_crop(image, labels, bboxes, min_object_covered, aspect_ratio_range):
    area_range = (0.1, 1.0)
    max_attempts = 200
    clip_bboxes = True
    with tf.name_scope('distorted_bounding_box_crop'):
        # Each bounding box has shape [1, num_boxes, box coords] and
        # the coordinates are ordered [ymin, xmin, ymax, xmax].

        # generate distorted bbox
        bbox_begin, bbox_size, distort_bbox = tf.image.sample_distorted_bounding_box(
                tf.shape(image),
                bounding_boxes=tf.expand_dims(bboxes, 0),
                min_object_covered=min_object_covered,
                aspect_ratio_range=aspect_ratio_range,
                area_range=area_range,
                max_attempts=max_attempts,
                use_image_if_no_bounding_boxes=True
        )

        distort_bbox = distort_bbox[0, 0]  # 1x1x4 --> 4
        cropped_image = tf.slice(image, bbox_begin, bbox_size)
        cropped_image.set_shape([None, None, 3])
        bboxes = bboxes_resize(distort_bbox, bboxes)
        labels, bboxes = bboxes_filter_overlap(labels, bboxes,
                                               threshold=BBOX_CROP_OVERLAP)

    return cropped_image, labels, bboxes, distort_bbox


def image_whitened(image, means):
    mean = tf.constant(means, dtype=image.dtype)
    image = image - mean
    return image


def image_unwhitened(image, means, to_int=True):
    mean = tf.constant(means, dtype=image.dtype)
    image = image + mean
    if to_int:
        image = tf.cast(image, tf.int32)
    return image
