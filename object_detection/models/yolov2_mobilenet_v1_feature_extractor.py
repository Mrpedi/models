"""
YOLOv2 Feature Extractor using DarkNet Features
"""

import tensorflow as tf

from object_detection.meta_architectures import yolov2_meta_arch
from object_detection.models import feature_map_generators
from nets import mobilenet_v1

slim = tf.contrib.slim


class YOLOv2DarkNetFeatureExtractor(yolov2_meta_arch.YOLOv2FeatureExtractor):
    """
    YOLOv2 Feature Extractor using MobileNet v1
    """
    def __init__(self, is_training,
                 reuse_weights=None):
        """
        :param depth_multiplier:
        :param min_depth:
        :param conv_hyperarams:
        :param reuse_weights:
        """
        super(YOLOv2DarkNetFeatureExtractor, self).__init__(is_training, reuse_weights)

    def preprocess(self, resized_inputs):
        return (resized_inputs / 255.0) * 2.0 - 1.0

    def _extract_features(self, preprocessed_inputs):
        pass

