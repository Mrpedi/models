from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from slim.nets import darknet

slim = tf.contrib.slim


class DarkNetTest(tf.test.TestCase):
  def testBuildClasssifcationNetwork(self):
    batch_size = 5
    height, width, channels = 608, 608, 3
    num_classes = 1000

    inputs = tf.random_uniform((batch_size, height, width, channels))
    logits, end_points = darknet.darknet_19(inputs, num_classes)
    self.assertTrue(logits.op.name.startswith('darknet_19/logits'))
    self.assertListEqual(logits.get_shape().as_list(),
                         [batch_size, num_classes])

  def testBuildBaseNetwork(self):
    batch_size = 5
    height, width, channels = 608, 608, 3

    inputs = tf.random_uniform((batch_size, height, width, channels))
    net, end_points = darknet.darknet_19_base(inputs)
    self.assertTrue(net.op.name.startswith('darknet_19/conv2d_18'))

  def testHalfSizeImages(self):
    batch_size = 5
    height, width, channels = 300, 300, 3
    num_classes = 1000

    inputs = tf.random_uniform((batch_size, height, width, channels))
    logits, end_points = darknet.darknet_19(inputs, num_classes)
    self.assertTrue(logits.op.name.startswith('darknet_19/logits'))
    self.assertListEqual(logits.get_shape().as_list(),
                         [batch_size, num_classes])

  def testUnknownImageShape(self):
    tf.reset_default_graph()
    batch_size = 2
    height, width, channels = 608, 608, 3
    num_classes = 1000

    input_np = np.random.uniform(0, 1, (batch_size, height, width, channels))
    with self.test_session() as sess:
      inputs = tf.placeholder(dtype=tf.float32, shape=(batch_size, None, None, 3))
      logits, end_points = darknet.darknet_19(inputs, num_classes)

      feed_dict = {inputs: input_np}
      tf.global_variables_initializer().run()

      predictions = sess.run(logits, feed_dict=feed_dict)

      self.assertTrue(logits.op.name.startswith('darknet_19/logits'))
      self.assertEquals(predictions.shape, (batch_size, num_classes))

  def testUnknownBatchSize(self):
    tf.reset_default_graph()
    batch_size = 1
    height, width, channels = 608, 608, 3
    num_classes = 1000

    inputs = tf.placeholder(dtype=tf.float32, shape=(None, height, width, channels))
    logits, end_points = darknet.darknet_19(inputs, num_classes)
    self.assertTrue(logits.op.name.startswith('darknet_19/logits'))
    self.assertListEqual(logits.get_shape().as_list(),
                         [None, num_classes])
    images = tf.random_uniform((batch_size, height, width, channels))

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      predictions = sess.run(logits, feed_dict={inputs: images.eval()})
      self.assertEqual(predictions.shape, (batch_size, num_classes))

  def testEvaluation(self):
    batch_size = 2
    height, width, channels = 608, 608, 3
    num_classes = 1000

    eval_inputs = tf.random_uniform((batch_size, height, width, channels))
    logits, end_points = darknet.darknet_19(eval_inputs, num_classes, is_training=False)

    predictions = tf.argmax(logits, 1)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      outputs = sess.run(predictions)
      self.assertEqual(outputs.shape, (batch_size,))

  def testTrainingEvalWithReuse(self):
    train_batch = 5
    eval_batch = 2
    height, width, channels = 608, 608, 3
    num_classes = 1000

    train_inputs = tf.random_uniform(shape=(train_batch, height, width, channels))
    eval_inputs = tf.random_uniform(shape=(eval_batch, height, width, channels))

    output, end_points = darknet.darknet_19(train_inputs, num_classes)
    logits, end_points = darknet.darknet_19(eval_inputs, num_classes, reuse=True)
    for i in end_points:
      print(i)
    predictions = tf.argmax(logits, 1)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      result = sess.run(predictions)
      self.assertEquals(result.shape, (eval_batch,))


if __name__ == '__main__':
  tf.test.main()
