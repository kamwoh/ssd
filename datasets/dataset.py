import os
import tensorflow as tf
from tensorflow.contrib import slim

FILE_PATTERN = 'voc_2007_%s_*.tfrecord'
ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'shape': 'Shape of the image',
    'object/bbox': 'A list of bounding boxes, one per each object.',
    'object/label': 'A list of labels, one per each object.',
}
# (Images, Objects) statistics on every class.
TRAIN_STATISTICS = {
    'none': (0, 0),
    'aeroplane': (238, 306),
    'bicycle': (243, 353),
    'bird': (330, 486),
    'boat': (181, 290),
    'bottle': (244, 505),
    'bus': (186, 229),
    'car': (713, 1250),
    'cat': (337, 376),
    'chair': (445, 798),
    'cow': (141, 259),
    'diningtable': (200, 215),
    'dog': (421, 510),
    'horse': (287, 362),
    'motorbike': (245, 339),
    'person': (2008, 4690),
    'pottedplant': (245, 514),
    'sheep': (96, 257),
    'sofa': (229, 248),
    'train': (261, 297),
    'tvmonitor': (256, 324),
    'total': (5011, 12608),
}
TEST_STATISTICS = {
    'none': (0, 0),
    'aeroplane': (1, 1),
    'bicycle': (1, 1),
    'bird': (1, 1),
    'boat': (1, 1),
    'bottle': (1, 1),
    'bus': (1, 1),
    'car': (1, 1),
    'cat': (1, 1),
    'chair': (1, 1),
    'cow': (1, 1),
    'diningtable': (1, 1),
    'dog': (1, 1),
    'horse': (1, 1),
    'motorbike': (1, 1),
    'person': (1, 1),
    'pottedplant': (1, 1),
    'sheep': (1, 1),
    'sofa': (1, 1),
    'train': (1, 1),
    'tvmonitor': (1, 1),
    'total': (20, 20),
}
SPLITS_TO_SIZES = {
    'train': 5011,
    'test': 4952,
}
SPLITS_TO_STATISTICS = {
    'train': TRAIN_STATISTICS,
    'test': TEST_STATISTICS,
}
NUM_CLASSES = 20


def get_dataset_provider(dataset, num_readers, batch_size):
    with tf.device('/device:CPU:0'):
        with tf.name_scope('pascal_voc_data_provider'):
            provider = slim.dataset_data_provider.DatasetDataProvider(dataset,
                                                                      num_readers=num_readers,
                                                                      common_queue_capacity=20 * batch_size,
                                                                      common_queue_min=10 * batch_size,
                                                                      shuffle=True)
        return provider


def get_dataset(dataset_dir, stage):
    assert stage.lower() in ['train', 'test']
    file_pattern = os.path.join(dataset_dir, FILE_PATTERN % stage)

    reader = tf.TFRecordReader
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/height': tf.FixedLenFeature([1], tf.int64),
        'image/width': tf.FixedLenFeature([1], tf.int64),
        'image/channels': tf.FixedLenFeature([1], tf.int64),
        'image/shape': tf.FixedLenFeature([3], tf.int64),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64),
        'image/object/bbox/difficult': tf.VarLenFeature(dtype=tf.int64),
        'image/object/bbox/truncated': tf.VarLenFeature(dtype=tf.int64)
    }
    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
        'shape': slim.tfexample_decoder.Tensor('image/shape'),
        'object/bbox': slim.tfexample_decoder.BoundingBox(['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/bbox/'),
        'object/label': slim.tfexample_decoder.Tensor('image/object/bbox/label'),
        'object/difficult': slim.tfexample_decoder.Tensor('image/object/bbox/difficult'),
        'object/truncated': slim.tfexample_decoder.Tensor('image/object/bbox/truncated')
    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

    labels_to_names = None

    if tf.gfile.Exists(os.path.join(dataset_dir, 'labels.txt')):
        labels_filename = os.path.join(dataset_dir, 'labels.txt')
        with tf.gfile.Open(labels_filename, 'rb') as f:
            lines = f.read()
        lines = lines.split(b'\n')
        lines = filter(None, lines)
        labels_to_names = {}
        for line in lines:
            index = line.index(b':')
            labels_to_names[int(line[:index])] = line[index + 1:]

    data_sources = ['%s/%s' % (dataset_dir, filename) for filename in os.listdir(dataset_dir)]

    return slim.dataset.Dataset(data_sources=data_sources,
                                reader=reader,
                                decoder=decoder,
                                num_samples=SPLITS_TO_SIZES[stage],
                                items_to_descriptions=ITEMS_TO_DESCRIPTIONS,
                                num_classes=NUM_CLASSES,
                                labels_to_names=labels_to_names)
