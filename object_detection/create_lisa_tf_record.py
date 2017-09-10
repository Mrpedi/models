# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Convert the LISA Traffic Sign dataset to TFRecord for object_detection.
See: https://cvrr.ucsd.edu/vivachallenge/index.php/signs/sign-detection/

Example usage:
    ./create_lisa_tf_record.py \
        --data_dir=/home/user/lisa/training \
        --output_dir=/home/user/lisa/output
"""

import hashlib
import io
import logging
import os
import random
import glob
import csv
from itertools import islice

import PIL.Image
import tensorflow as tf

from utils import dataset_util
from utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to LISA Extension dataset.')
flags.DEFINE_string('output_dir', '', 'Path to directory to output TFRecords.')
flags.DEFINE_string('label_map_path', 'data/lisa_label_map.pbtxt',
                    'Path to label map proto')
FLAGS = flags.FLAGS
FIELD_NAMES = ['Filename', 'Annotation tag', 'Upper left corner X', 'Upper left corner Y', 'Lower right corner X',
               'Lower right corner Y']


def parse_lisa_annotations_to_dict(data_dir):
    """

    :param data_dir:
    :return:
    """
    global FIELD_NAMES
    # Assume LISA Dataset only have one annotation
    csv_file = glob.glob(data_dir + "*.csv")[0]

    # Extract bounding boxes from training data
    training_instances = {}

    with open(csv_file) as f:
        reader = csv.DictReader(f, fieldnames=FIELD_NAMES, delimiter=';')
        for row in islice(reader, 1, None):  # skip header file
            img_path = row['Filename']
            category = row['Annotation tag']
            xmin = row['Upper left corner X']
            ymin = row['Upper left corner Y']
            xmax = row['Lower right corner X']
            ymax = row['Lower right corner Y']

            # Define a new object dictionary following PASCAL standard
            an_object = {'category': category,
                         'bndbox': {'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax},
                         'difficult': 1,
                         'truncated': 0,
                         'pose': ''}

            # Generate key using image path
            key = hashlib.sha256(img_path).hexdigest()

            if key in training_instances:
                training_instances[key]['object'].append(an_object)

            else:
                training_instances[key] = {'filename': img_path,
                                           'object': [an_object]}

    return training_instances


class ImageCoder(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self, sess):
    # Create a single Session to run all image coding calls.
    self._sess = sess

    # Initializes function that converts PNG to JPEG data.
    self._png_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_png(self._png_data, channels=3)
    self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def png_to_jpeg(self, image_data):
    return self._sess.run(self._png_to_jpeg,
                          feed_dict={self._png_data: image_data})

  def decode_jpeg(self, image_data):
    image = self._sess.run(self._decode_jpeg,
                           feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


def dict_to_tf_example(data,
                       label_map_dict, image_coder,
                       ignore_difficult_instances=False):
    """Convert XML derived dict to tf.Example proto.
    Notice that this function normalizes the bounding box coordinates provided
    by the raw data.
    Args:
      data: dict holding PASCAL XML fields for a single image (obtained by
        running dataset_util.recursive_parse_xml_to_dict)
      label_map_dict: A map from string label names to integers ids.
      ignore_difficult_instances: Whether to skip difficult instances in the
        dataset  (default: False).

      image_coder: ImageCoder Object - Utility to convert PNG image to JPEG

    Returns:
      example: The converted tf.Example.
    Raises:
      ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """
    img_path = os.path.join(FLAGS.data_dir, data['filename'])
    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_png = fid.read()

        # Decode image from PNG to JPEG format
        encoded_jpg = image_coder.png_to_jpeg(encoded_png)

    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    # Open image and find its size
    with PIL.Image.open(img_path) as img:
        width, height = img.size

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []
    for obj in data['object']:
        difficult = bool(int(obj['difficult']))
        if ignore_difficult_instances and difficult:
            continue

        difficult_obj.append(int(difficult))
        xmin.append(float(obj['bndbox']['xmin']) / width)
        ymin.append(float(obj['bndbox']['ymin']) / height)
        xmax.append(float(obj['bndbox']['xmax']) / width)
        ymax.append(float(obj['bndbox']['ymax']) / height)
        class_name = obj['category']
        classes_text.append(class_name.encode('utf8'))
        classes.append(label_map_dict[class_name])
        truncated.append(int(obj['truncated']))
        poses.append(obj['pose'].encode('utf8'))

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(data['filename'].encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(data['filename'].encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
        'image/object/truncated': dataset_util.int64_list_feature(truncated),
        'image/object/view': dataset_util.bytes_list_feature(poses),
    }))
    return example


def create_tf_record(output_filename,
                     label_map_dict,
                     examples, image_coder):
    """Creates a TFRecord file from examples.
    Args:
      output_filename: Path to where output file is saved.
      label_map_dict: The label map dictionary.
      examples: Examples to parse and save to tf record.
      image_coder: Utility to convert PNG image to JPEG
    """
    writer = tf.python_io.TFRecordWriter(output_filename)
    for idx, example in enumerate(examples):
        if idx % 100 == 0:
            logging.info('On image %d of %d', idx, len(examples))
            print 'On image %d of %d', idx, len(examples)
        data = example
        tf_example = dict_to_tf_example(data, label_map_dict,image_coder)
        writer.write(tf_example.SerializeToString())

    writer.close()


# TODO: Add test for pet/PASCAL main files.
def main(_):
    data_dir = FLAGS.data_dir
    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

    logging.info('Reading from LISA Extension dataset.')

    examples = parse_lisa_annotations_to_dict(data_dir)
    examples_list = [i[1] for i in examples.items()]    # Convert dict to list

    # Test images are not included in the downloaded data set, so we shall perform
    # our own split.
    random.seed(42)
    random.shuffle(examples_list)
    num_examples = len(examples_list)
    num_train = int(0.8 * num_examples)
    train_examples = examples_list[:num_train]
    val_examples = examples_list[num_train:]
    logging.info('%d training and %d validation examples.',
                 len(train_examples), len(val_examples))
    print('%d training and %d validation examples.',
                 len(train_examples), len(val_examples))

    train_output_path = os.path.join(FLAGS.output_dir, 'lisa_train.record')
    val_output_path = os.path.join(FLAGS.output_dir, 'lisa_val.record')

    # Create a session to encode/decode PNG -> JPEG images
    sess = tf.Session()
    image_coder = ImageCoder(sess)

    create_tf_record(train_output_path, label_map_dict, train_examples, image_coder)
    create_tf_record(val_output_path, label_map_dict, val_examples, image_coder)

    sess.close()

if __name__ == '__main__':
    tf.app.run()
