'''
provide custom metrics
'''

# built-in
import pdb
import os
from multiprocessing import cpu_count

# external
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

# custom
from . import image as custom_image_ops


def solve_metric(metric_spec):
    '''
    solve metric spec and return metric instance
    '''
    if isinstance(metric_spec, str): return metric_spec
    elif isinstance(metric_spec, dict):
        assert len(metric_spec) == 1
        pass
    else: raise ValueError

    metric_name, metric_options = list(metric_spec.items())[0]
    instance = tf.keras.metrics.get({
        "class_name": metric_name,
        "config": metric_options,
    })
    return instance


class FBetaScore(tf.keras.metrics.Metric):
    def __init__(self, beta, thresholds, epsilon=1e-07, **kargs):
        super().__init__(**kargs)
        assert beta > 0
        self.beta = beta
        self.epsilon = epsilon
        self.thresholds = thresholds
        self.prepare_precision_recall()
        return

    def prepare_precision_recall(self):
        self.precision = tf.keras.metrics.Precision(thresholds=self.thresholds)
        self.recall = tf.keras.metrics.Recall(thresholds=self.thresholds)
        return

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight=sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight=sample_weight)
        return

    def result(self):
        precision = self.precision.result()
        recall = self.recall.result()
        score = (1 + self.beta**2) * precision * recall / (self.beta**2 * precision + recall + self.epsilon)
        return score

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()
        return

    def get_config(self):
        """Returns the serializable config of the metric."""
        config = super().get_config()
        config.update({'beta': self.beta, 'epsilon': self.epsilon, 'thresholds': self.thresholds})
        return config


class _RegionBasedMetric(tf.keras.metrics.Metric):
    '''abstract class to provide common methods for region based metrics

    Args:
        thresholds: scalar or vector of thresholds
        IoU_threshold: minimum IoU between prediction and label
            required to be considered "successful detected"
        epsilon: small number to avoid devision by zero
    '''
    def __init__(self, thresholds, IoU_threshold=0.30, epsilon=1e-07, **kargs):
        super().__init__(**kargs)
        with tf.control_dependencies([tf.debugging.assert_non_negative(thresholds)]):
            self.thresholds = thresholds
        self.IoU_threshold = IoU_threshold
        self.epsilon = epsilon
        return

    @tf.function
    def _separate_predictions(self, single_label, single_pred):
        '''separate positive(cancer) regions

        A mask contains multiple positive regions.
        This method will separate those regions and generate a separeted mask for each of them
        by applying connected components analysis.
        For example, if a mask has N cancer regions, then this single mask will be separated into N masks.

        Args:
            single_label: label mask. This should not be batched.
            single_pred: prediction mask. This should not be batched.
                A prediction should be already be thresholded.

        Returns:
            indiced_label: label mask for each cancer region.
            indiced_pred: prediction mask for each predicted cancer region.
        '''
        cca_label = tfa.image.connected_components(single_label)
        indiced_label = tf.one_hot(
            cca_label, tf.reduce_max(cca_label) + 1, axis=0, dtype=tf.bool, on_value=True, off_value=False)[1:]

        single_pred = tf.broadcast_to(
            tf.cast(single_pred, tf.float32),
            [tf.shape(self.thresholds)[0], tf.shape(single_pred)[0], tf.shape(single_pred)[1]],
        )
        single_pred = tf.transpose(tf.transpose(single_pred, [1, 2, 0]) > self.thresholds, [2, 0, 1])
        single_pred = tf.squeeze(custom_image_ops.morph_open(tf.expand_dims(tf.cast(single_pred, tf.int8), -1), 5), -1)
        cca_pred = tfa.image.connected_components(single_pred)
        # cca_pred dims: n_thresholds, H, W
        min_indices = -tf.sparse.reduce_max(tf.sparse.from_dense(-cca_pred), axis=[1, 2]) - 1
        should_shift = tf.cast(min_indices > 0, min_indices.dtype)
        substractor = tf.transpose(tf.broadcast_to(
            min_indices * should_shift,
            [tf.shape(single_pred)[1], tf.shape(single_pred)[2], tf.shape(self.thresholds)[0]],
        ), [2, 0, 1])
        substractor = tf.zeros_like(cca_pred) + substractor * tf.cast(cca_pred > 0, tf.int32)
        cca_pred = cca_pred - substractor
        # cca_pred dims: n_thresholds, H, W
        indiced_pred = tf.one_hot(
            cca_pred, tf.reduce_max(cca_pred) + 1, axis=0, dtype=tf.bool, on_value=True, off_value=False)[1:]
        # indiced_pred dims: masks, n_thresholds, H, W
        indiced_pred = tf.transpose(indiced_pred, [0, 2, 3, 1])
        # indiced_pred dims: masks, H, W, thresholds
        lengths = tf.reduce_any(indiced_pred, axis=[1, 2])
        if tf.shape(indiced_pred)[0] > 0:
            lengths = tf.argmin(tf.cast(lengths, tf.uint8), axis=0) + 1
        else:
            lengths = tf.zeros(tf.shape(self.thresholds), tf.int64)
        return indiced_label, indiced_pred, lengths

    @tf.function
    def _IoU(self, indiced_label, indiced_pred):
        '''
        given multiple label cancer region masks and multiple predicted cancer region masks,
        this method will calculate IoU.

        Args:
            indiced_label: label cancer masks
                shape: [N_masks, height, width], dtype: tf.bool
            cancer_pred: single cancer prediction mask
                shape: [M_masks, height, width, N_thresholds], dtype: tf.bool

        Returns:
            IoU vector
                shape: [N_masks, M_masks, N_thresholds]
        '''
        n_label_mask, n_pred_mask = tf.shape(indiced_label)[0], tf.shape(indiced_pred)[0]
        n_thresholds = tf.shape(self.thresholds)[0]
        intermediate_shape = [n_label_mask, n_pred_mask, tf.shape(indiced_label)[1], tf.shape(indiced_label)[2], n_thresholds]

        indiced_label = tf.transpose(tf.broadcast_to(
            indiced_label,
            [n_pred_mask, n_thresholds, n_label_mask, tf.shape(indiced_label)[1], tf.shape(indiced_label)[2]],
        ), [2, 0, 3, 4, 1])
        indiced_pred = tf.broadcast_to(indiced_pred, intermediate_shape)
        intersection = tf.reduce_sum(tf.cast(indiced_label & indiced_pred, tf.float32), axis=[2, 3])
        union = tf.reduce_sum(tf.cast(indiced_label | indiced_pred, tf.float32), axis=[2, 3])
        iou = intersection / union
        return iou

    @tf.function
    def get_tp_fn(self, y_true, y_pred, sample_weight):
        if sample_weight is not None: raise NotImplementedError
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.squeeze(y_pred, -1)
        y_true_pred = tf.cast(tf.stack([y_true, y_pred], axis=1), tf.float32)

        tp_array, fn_array = tf.map_fn(
            lambda single_label_pred: self.get_label_detected(single_label_pred[0], single_label_pred[1]),
            y_true_pred,
            fn_output_signature=(tf.int32, tf.int32),
            parallel_iterations=cpu_count(),
        )
        tp = tf.reduce_sum(tp_array, axis=0)
        fn = tf.reduce_sum(fn_array, axis=0)
        return tp, fn

    @tf.function
    def get_label_detected(self, single_label, single_pred):
        single_label = tf.cast(single_label, tf.bool)
        indiced_label, indiced_pred, pred_masks = self._separate_predictions(single_label, single_pred)

        IoU_matrix = self._IoU(indiced_label, indiced_pred)
        label_detected = tf.reduce_any(IoU_matrix > self.IoU_threshold, axis=1)

        tp = tf.reduce_sum(tf.cast(label_detected, tf.int32), axis=0)
        fn = tf.reduce_sum(tf.cast(~label_detected, tf.int32), axis=0)
        return tp, fn

    @tf.function
    def get_tp_fp(self, y_true, y_pred, sample_weight):
        if sample_weight is not None: raise NotImplementedError
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.squeeze(y_pred, -1)
        y_true_pred = tf.cast(tf.stack([y_true, y_pred], axis=1), tf.float32)

        tp_array, fp_array = tf.map_fn(
            lambda single_label_pred: self.get_tp_pred(single_label_pred[0], single_label_pred[1]),
            y_true_pred,
            fn_output_signature=(tf.int32, tf.int32),
            parallel_iterations=cpu_count(),
        )
        tp = tf.reduce_sum(tp_array, axis=0)
        fp = tf.reduce_sum(fp_array, axis=0)
        return tp, fp

    @tf.function
    def get_tp_pred(self, single_label, single_pred):
        indiced_label, indiced_pred, n_pred_masks = self._separate_predictions(single_label, single_pred)

        IoU_matrix = self._IoU(indiced_label, indiced_pred)
        tp_pred = tf.reduce_any(IoU_matrix > self.IoU_threshold, axis=0)
        tp_pred = tf.RaggedTensor.from_tensor(tf.transpose(tp_pred), n_pred_masks)

        tp = tf.reduce_sum(tf.cast(tp_pred, tf.int32), axis=1)
        fp = tf.reduce_sum(tf.cast(~tp_pred, tf.int32), axis=1)
        return tp, fp

    def get_config(self):
        configs = super().get_config()
        configs['thresholds'] = self.thresholds
        configs['IoU_threshold'] = self.IoU_threshold
        configs['epsilon'] = self.epsilon
        return configs


class RegionBasedFBetaScore(FBetaScore):
    def __init__(self, beta, thresholds, IoU_threshold=0.30, epsilon=1e-07, **kargs):
        self.IoU_threshold = IoU_threshold
        super().__init__(beta=beta, thresholds=thresholds, epsilon=epsilon, **kargs)
        return

    def prepare_precision_recall(self):
        self.precision = RegionBasedPrecision(thresholds=self.thresholds, IoU_threshold=self.IoU_threshold, epsilon=self.epsilon)
        self.recall = RegionBasedRecall(thresholds=self.thresholds, IoU_threshold=self.IoU_threshold, epsilon=self.epsilon)
        return

    def get_config(self):
        """Returns the serializable config of the metric."""
        config = super().get_config()
        config.update({'IoU_threshold': self.IoU_threshold})
        return config


class RegionBasedRecall(_RegionBasedMetric):
    def __init__(self, thresholds, IoU_threshold=0.30, epsilon=1e-07, **kargs):
        thresholds = tf.reshape(thresholds, [-1])
        super().__init__(thresholds, IoU_threshold, epsilon, **kargs)
        self.tp_count = self.add_weight(
            'tp_count', dtype=tf.int32, shape=tf.shape(self.thresholds), initializer=tf.zeros_initializer)
        self.fn_count = self.add_weight(
            'fn_count', dtype=tf.int32, shape=tf.shape(self.thresholds), initializer=tf.zeros_initializer)
        return

    def reset_states(self):
        self.tp_count.assign(tf.zeros_like(self.tp_count))
        self.fn_count.assign(tf.zeros_like(self.fn_count))
        return

    @tf.function
    def update_state(self, y_true, y_pred, sample_weight=None):
        tp, fn = self.get_tp_fn(y_true, y_pred, sample_weight)
        self.tp_count.assign_add(tp)
        self.fn_count.assign_add(fn)
        return

    def result(self):
        result = tf.cast(self.tp_count, tf.float32) / (tf.cast(self.tp_count + self.fn_count, tf.float32) + self.epsilon)
        result = tf.squeeze(result)
        return result


class RegionBasedPrecision(_RegionBasedMetric):
    def __init__(self, thresholds, IoU_threshold=0.30, epsilon=1e-07, **kargs):
        thresholds = tf.reshape(thresholds, [-1])
        super().__init__(thresholds, IoU_threshold, epsilon, **kargs)
        self.tp_count = self.add_weight(
            'tp_count', dtype=tf.int32, shape=tf.shape(self.thresholds), initializer=tf.zeros_initializer)
        self.fp_count = self.add_weight(
            'fp_count', dtype=tf.int32, shape=tf.shape(self.thresholds), initializer=tf.zeros_initializer)
        return

    def reset_states(self):
        self.tp_count.assign(tf.zeros_like(self.tp_count))
        self.fp_count.assign(tf.zeros_like(self.fp_count))
        return

    @tf.function
    def update_state(self, y_true, y_pred, sample_weight=None):
        tp, fp = self.get_tp_fp(y_true, y_pred, sample_weight)
        self.tp_count.assign_add(tp)
        self.fp_count.assign_add(fp)
        return

    def result(self):
        result = tf.cast(self.tp_count, tf.float32) / (tf.cast(self.tp_count + self.fp_count, tf.float32) + self.epsilon)
        result = tf.squeeze(result)
        return result


class RegionBasedTruePositives(_RegionBasedMetric):
    def __init__(self, thresholds, IoU_threshold=0.30, epsilon=1e-07, **kargs):
        thresholds = tf.reshape(thresholds, [-1])
        super().__init__(thresholds, IoU_threshold, epsilon, **kargs)
        self.tp_count = self.add_weight(
            'tp_count', dtype=tf.int32, shape=tf.shape(self.thresholds), initializer=tf.zeros_initializer)
        return

    def reset_states(self):
        self.tp_count.assign(tf.zeros_like(self.tp_count))
        return

    @tf.function
    def update_state(self, y_true, y_pred, sample_weight=None):
        tp, fn = self.get_tp_fn(y_true, y_pred, sample_weight)
        self.tp_count.assign_add(tp)
        return

    def result(self):
        result = self.tp_count
        result = tf.squeeze(result)
        return result


class RegionBasedFalsePositives(_RegionBasedMetric):
    def __init__(self, thresholds, IoU_threshold=0.30, epsilon=1e-07, **kargs):
        thresholds = tf.reshape(thresholds, [-1])
        super().__init__(thresholds, IoU_threshold, epsilon, **kargs)
        self.fp_count = self.add_weight(
            'fp_count', dtype=tf.int32, shape=tf.shape(self.thresholds), initializer=tf.zeros_initializer)
        return

    def reset_states(self):
        self.fp_count.assign(tf.zeros_like(self.fp_count))
        return

    @tf.function
    def update_state(self, y_true, y_pred, sample_weight=None):
        tp, fp = self.get_tp_fp(y_true, y_pred, sample_weight)
        self.fp_count.assign_add(fp)
        return

    def result(self):
        result = self.fp_count
        result = tf.squeeze(result)
        return result


class RegionBasedFalseNegatives(_RegionBasedMetric):
    def __init__(self, thresholds, IoU_threshold=0.30, epsilon=1e-07, **kargs):
        thresholds = tf.reshape(thresholds, [-1])
        super().__init__(thresholds, IoU_threshold, epsilon, **kargs)
        self.fn_count = self.add_weight(
            'fn_count', dtype=tf.int32, shape=tf.shape(self.thresholds), initializer=tf.zeros_initializer)
        return

    def reset_states(self):
        self.fn_count.assign(tf.zeros_like(self.fn_count))
        return

    @tf.function
    def update_state(self, y_true, y_pred, sample_weight=None):
        tp, fn = self.get_tp_fn(y_true, y_pred, sample_weight)
        self.fn_count.assign_add(fn)
        return

    def result(self):
        result = self.fn_count
        result = tf.squeeze(result)
        return result


class DummyMetric():
    def __init__(self, value):
        self.value = value
        return

    def update_state(self, *args, **kargs):
        return

    def result(self):
        return self.value


tf.keras.utils.get_custom_objects().update(FBetaScore=FBetaScore)
tf.keras.utils.get_custom_objects().update(RegionBasedRecall=RegionBasedRecall)
tf.keras.utils.get_custom_objects().update(RegionBasedPrecision=RegionBasedPrecision)
tf.keras.utils.get_custom_objects().update(RegionBasedFBetaScore=RegionBasedFBetaScore)
tf.keras.utils.get_custom_objects().update(RegionBasedTruePositives=RegionBasedTruePositives)
tf.keras.utils.get_custom_objects().update(RegionBasedFalsePositives=RegionBasedFalsePositives)
tf.keras.utils.get_custom_objects().update(RegionBasedFalseNegatives=RegionBasedFalseNegatives)
