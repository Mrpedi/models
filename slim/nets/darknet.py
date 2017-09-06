"""
Model Definition of DarkNet-19, used in You Only Look Once v2 Object Detection Model

For more information : https://pjreddie.com/darknet/imagenet/

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim


def leaky_relu(x, alpha=0.01):
  """
  Leaky Rectified Linear Unit (Leaky-ReLU) activation Function
  """
  return tf.maximum(alpha * x, x)


def darknet_19_arg_scope(is_training=True,
                         weight_decay=5e-4):
  """
  DarkNet TF-slim Argument Scope
  """
  batch_morm_params = {
    'is_training': is_training,
    'center': True,
    'scale': True,
    'decay': 0.9997,
    'epsilon': 0.001,
  }
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=leaky_relu,
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      normalizer_fn=slim.batch_norm):
    with slim.arg_scope([slim.batch_norm], **batch_morm_params):
      with slim.arg_scope([slim.conv2d], padding='SAME') as arg_scope:
        return arg_scope


def darknet_19_base(inputs, reuse=None, scope=None):
  """
  Construct DarkNet base feature extractor (without logits).
  Args:
      inputs: a tf.float32 Tensor
      reuse:
      scope:

  Returns:
      net:
      end_points: set of layers
  """

  end_points = {}
  with tf.variable_scope(scope, 'DarkNet_19', [inputs], reuse=reuse) as sc:
    end_points_collection = sc.name
    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.batch_norm],
                        outputs_collections=end_points_collection):
      net = slim.conv2d(inputs, 32, [3, 3], scope='Conv2d_1', normalizer_fn=slim.batch_norm)
      net = slim.max_pool2d(net, [2, 2], scope='MaxPool1')

      net = slim.conv2d(net, 64, [3, 3], scope='Conv2d_2', normalizer_fn=slim.batch_norm)
      net = slim.max_pool2d(net, [2, 2], scope='MaxPool2')

      net = slim.conv2d(net, 128, [3, 3], scope='Conv2d_3', normalizer_fn=slim.batch_norm)
      net = slim.conv2d(net, 64,  [1, 1], scope='Conv2d_4', normalizer_fn=slim.batch_norm)
      net = slim.conv2d(net, 128, [3, 3], scope='Conv2d_5', normalizer_fn=slim.batch_norm)
      net = slim.max_pool2d(net,  [2, 2], scope='MaxPool3')

      net = slim.conv2d(net, 256, [3, 3], scope='Conv2d_6', normalizer_fn=slim.batch_norm)
      net = slim.conv2d(net, 128, [1, 1], scope='Conv2d_7', normalizer_fn=slim.batch_norm)
      net = slim.conv2d(net, 256, [3, 3], scope='Conv2d_8', normalizer_fn=slim.batch_norm)
      net = slim.max_pool2d(net,  [2, 2], scope='MaxPool4')

      net = slim.conv2d(net, 512, [3, 3], scope='Conv2d_9', normalizer_fn=slim.batch_norm)
      net = slim.conv2d(net, 256, [1, 1], scope='Conv2d_10', normalizer_fn=slim.batch_norm)
      net = slim.conv2d(net, 512, [3, 3], scope='Conv2d_11', normalizer_fn=slim.batch_norm)
      net = slim.conv2d(net, 256, [1, 1], scope='Conv2d_12', normalizer_fn=slim.batch_norm)
      net = slim.conv2d(net, 512, [3, 3], scope='Conv2d_13', normalizer_fn=slim.batch_norm)
      net = slim.max_pool2d(net,  [2, 2], scope='MaxPool4')

      net = slim.conv2d(net, 1024, [3, 3], scope='Conv2d_14', normalizer_fn=slim.batch_norm)
      net = slim.conv2d(net, 512,  [1, 1], scope='Conv2d_15', normalizer_fn=slim.batch_norm)
      net = slim.conv2d(net, 1024, [3, 3], scope='Conv2d_16', normalizer_fn=slim.batch_norm)
      net = slim.conv2d(net, 512,  [1, 1], scope='Conv2d_17', normalizer_fn=slim.batch_norm)
      net = slim.conv2d(net, 1024, [3, 3], scope='Conv2d_18')

      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      return net, end_points


def darknet_19(inputs, num_classes=1000, is_training=True, reuse=None, scope='DarkNet_19'):
  """
  Darknet-19 Architecture

  Args:
    inputs:
    num_classes:
    is_training:
    reuse:
    scope:

  Return:

  """
  with tf.variable_scope(scope, 'DarkNet_19', [inputs, num_classes], reuse=reuse) as scope:
    with slim.arg_scope([slim.batch_norm], is_training=is_training):
      net, end_points = darknet_19_base(inputs, scope=scope)
      with tf.variable_scope('Logits'):
        net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None)
        with tf.variable_scope('global_avg_pool'):
          net    = slim.avg_pool2d(net, [1, 1])
          logits = tf.reduce_mean(net, [1, 2])
        end_points['Logits'] = logits
        end_points['Predictions'] = slim.softmax(logits)
      return logits, end_points

