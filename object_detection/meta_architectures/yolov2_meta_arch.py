"""
YOLOv2 Architecture Definition

General Tensorflow Implementation of You Only Look Once (YOLOv2) detection model

Reference: https://arxiv.org/abs/1612.08242
"""
from abc import abstractmethod
import tensorflow as tf

from object_detection.core import model
from object_detection.core import box_list
slim = tf.contrib.slim


class YOLOv2FeatureExtractor(object):
    """YOlO v2 Feature Extractor"""

    def __init__(self, is_training, reuse_weights=None):
        """
        Constructor
        :param is_training:  a boolean -
        :param reuse_weights:
        """
        self._is_training = is_training
        self._reuse_weights = reuse_weights

    @abstractmethod
    def preprocess(self, resized_inputs):
        """
        Pre-process image input for feature extraction ((minus image resizing)

        Args:
            resized_inputs: a float tensor - represents batch of resized images
                              [batch, height, width, channels]
        Return:
            preprocessed_inputs: a float tensor - batch of preprocessed images
        """
        pass

    @abstractmethod
    def _extract_features(self, preprocessed_inputs):
        """
        Output of feature extractor

        Args:
            preprocessed_inputs:

        Returns:

        """
        pass


class YOLOv2MetaArch(model.DetectionModel):
    """
    YOLOv2 Meta-Architecture Definition
    """
    def __init__(self,
                 is_training,
                 anchor_generator,
                 box_predictor,
                 feature_extractor,
                 matcher,
                 region_similarity_calculator,
                 image_resizer_fn,
                 non_max_suppression_fn,
                 score_conversion_fn,
                 localization_loss,
                 classification_loss,
                 localization_loss_weight,
                 classification_loss_weight,
                 add_summaries=True):
        super(YOLOv2MetaArch, self).__init__(num_classes=box_predictor.num_classes)

        self._is_training = is_training

        self._feature_tractor  = feature_extractor
        self._anchor_generator = anchor_generator
        self._matcher          = matcher

        self._image_resizer_fn      = image_resizer_fn
        self._non_max_suppression_fn = non_max_suppression_fn
        self._score_conversion_fn   = score_conversion_fn

        self._classification_loss = classification_loss
        self._localization_loss   = localization_loss
        self._classification_loss_weight = classification_loss_weight
        self._localization_loss_weight   = localization_loss_weight

        self._anchors = None
        self._add_summaries =  add_summaries

    @property
    def anchors(self):
        """
        Think of anchors as scaling factors for width/height outputs to become full-sized bounding boxes.
        :return:
        """
        if not self._anchors:
            raise RuntimeError('Anchors have not been constructed yet!')
        if not isinstance(self._anchors, box_list.BoxList):
            raise RuntimeError('Anchor should be a BoxList object, but it is currently not.')
        return self._anchors

    def preprocess(self, inputs):
        """"
        Feature-extractor specific preprocessing.

        Args:
          inputs: a [batch, height_in, width_in, channels] float tensor representing
            a batch of images with values between 0 and 255.0.

        Returns:
          preprocessed_inputs: a [batch, height_out, width_out, channels] float
            tensor representing a batch of images.
        Raises:
          ValueError: if inputs tensor does not have type tf.float32
        """
        if inputs.dtype is not tf.float32:
            raise ValueError('`preprocess` expects a tf.float32 tensor')
        with tf.name_scope('Preprocessor'):
            resized_inputs = tf.map_fn(self._image_resizer_fn,
                                       elems=inputs,
                                       dtype=tf.float32)

            return self._feature_tractor.preprocess(resized_inputs)

    def predict(self, preprocessed_inputs):
        raise NotImplemented

    def postprocess(self, prediction_dict, **params):
        raise  NotImplemented

    def loss(self, prediction_dict):
        raise NotImplemented

    def restore_map(self, from_detection_checkpoint=True):
        raise NotImplemented
