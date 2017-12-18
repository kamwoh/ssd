import tensorflow as tf
import os
import random
import sys
import xml.etree.ElementTree as ET

VOC_LABELS = {
    'none': (0, 'Background'),
    'aeroplane': (1, 'Vehicle'),
    'bicycle': (2, 'Vehicle'),
    'bird': (3, 'Animal'),
    'boat': (4, 'Vehicle'),
    'bottle': (5, 'Indoor'),
    'bus': (6, 'Vehicle'),
    'car': (7, 'Vehicle'),
    'cat': (8, 'Animal'),
    'chair': (9, 'Indoor'),
    'cow': (10, 'Animal'),
    'diningtable': (11, 'Indoor'),
    'dog': (12, 'Animal'),
    'horse': (13, 'Animal'),
    'motorbike': (14, 'Vehicle'),
    'person': (15, 'Person'),
    'pottedplant': (16, 'Indoor'),
    'sheep': (17, 'Animal'),
    'sofa': (18, 'Indoor'),
    'train': (19, 'Vehicle'),
    'tvmonitor': (20, 'Indoor'),
}


def to_feature(value, feature_type):
    if not isinstance(value, list):
        value = [value]

    if feature_type == 'int64':
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
    elif feature_type == 'float':
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))
    else:
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def main():
    dirname = os.path.dirname(__file__)

    dataset_dir = '%s/VOCdevkit_train/VOC2007' % dirname
    image_dir = '%s/JPEGImages' % dataset_dir
    anno_dir = '%s/Annotations' % dataset_dir

    output_dir = '%s/VOC_SSD' % dirname

    if not tf.gfile.Exists(output_dir):
        tf.gfile.MakeDirs(output_dir)

    filenames = os.listdir(anno_dir)  # avoid image with no annotation

    random.seed(829)
    random.shuffle(filenames)

    i = 0
    fidx = 0
    while i < len(filenames):
        tf_filename = '%s/%s_%03d.tfrecord' % (output_dir, 'voc_train', fidx)

        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            j = 0
            while i < len(filenames) and j < 200:
                sys.stdout.write('\r>> Converting image %d/%d' % (i + 1, len(filenames)))
                sys.stdout.flush()

                filename = filenames[i]
                img_name = os.path.splitext(filename)[0]

                # read image
                img_filename = '%s/%s.jpg' % (image_dir, img_name)
                img_data = tf.gfile.FastGFile(img_filename, 'r').read()

                # read annotation
                xml_filename = '%s/%s.xml' % (anno_dir, img_name)
                tree = ET.parse(xml_filename)
                root = tree.getroot()

                # image shape
                size = root.find('size')
                shape = [int(size.find('height').text),
                         int(size.find('width').text),
                         int(size.find('depth').text)]

                # find annotations
                bboxes = []
                labels = []
                labels_text = []
                difficult = []
                truncated = []
                for obj in root.findall('object'):
                    label = obj.find('name').text
                    labels.append(int(VOC_LABELS[label][0]))
                    labels_text.append(label.encode('ascii'))

                    if obj.find('difficult'):
                        difficult.append(int(obj.find('difficult').text))
                    else:
                        difficult.append(0)

                    if obj.find('truncated'):
                        truncated.append(int(obj.find('truncated').text))
                    else:
                        truncated.append(0)

                    bbox = obj.find('bndbox')
                    bboxes.append((float(bbox.find('ymin').text) / shape[0],
                                   float(bbox.find('xmin').text) / shape[1],
                                   float(bbox.find('ymax').text) / shape[0],
                                   float(bbox.find('xmax').text) / shape[1]))

                xmin = []
                ymin = []
                xmax = []
                ymax = []

                for b in bboxes:
                    y1, x1, y2, x2 = b

                    # python god algorithm!!
                    # [l.append(point) for l, point in zip([ymin, xmin, ymax, xmax], b)]

                    # naive algorithm :P
                    ymin.append(y1)
                    xmin.append(x1)
                    ymax.append(y2)
                    xmax.append(x2)

                img_format = b'JPEG'
                example = tf.train.Example(features=tf.train.Features(feature={
                    'image/height': to_feature(shape[0], 'int64'),
                    'image/width': to_feature(shape[1], 'int64'),
                    'image/channels': to_feature(shape[2], 'int64'),
                    'image/shape': to_feature(shape, 'int64'),
                    'image/object/bbox/xmin': to_feature(xmin, 'float'),
                    'image/object/bbox/xmax': to_feature(xmax, 'float'),
                    'image/object/bbox/ymin': to_feature(ymin, 'float'),
                    'image/object/bbox/ymax': to_feature(ymax, 'float'),
                    'image/object/bbox/label': to_feature(labels, 'int64'),
                    'image/object/bbox/label_text': to_feature(labels_text, 'bytes'),
                    'image/object/bbox/difficult': to_feature(difficult, 'int64'),
                    'image/object/bbox/truncated': to_feature(truncated, 'int64'),
                    'image/format': to_feature(img_format, 'bytes'),
                    'image/encoded': to_feature(img_data, 'bytes')
                }))

                tfrecord_writer.write(example.SerializeToString())

                i += 1
                j += 1

            fidx += 1
    print 'done!'

if __name__ == '__main__':
    main()