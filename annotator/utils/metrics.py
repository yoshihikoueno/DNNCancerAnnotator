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
        indiced_label = tf.one_hot(tfa.image.connected_components(single_label), tf.reduce_max(single_label) + 1)[:, :, 1:]
        indiced_label = tf.cast(tf.transpose(indiced_label, [2, 0, 1]), tf.bool)
        indiced_pred = tf.one_hot(tfa.image.connected_components(single_pred), tf.reduce_max(single_pred) + 1)[:, :, 1:]
        indiced_pred = tf.cast(tf.transpose(indiced_pred, [2, 0, 1]), tf.bool)
        return indiced_label, indiced_pred

    @tf.function
    def _IoU(self, cancer_label, cancer_pred):
        '''
        given single label cancer region and single predicted cancer region,
        this method will calculate IoU.
        '''
        intersection = tf.reduce_sum(tf.cast(cancer_label & cancer_pred, tf.float32))
        union = tf.reduce_sum(tf.cast(cancer_label | cancer_pred, tf.float32))
        iou = intersection / union
        return iou

    @tf.function
    def get_tp_fn(self, y_true, y_pred, sample_weight, threshold):
        if sample_weight is not None: raise NotImplementedError
        y_pred = tf.squeeze(tf.cast(y_pred > threshold, y_pred.dtype), -1)
        y_true_pred = tf.cast(tf.stack([y_true, y_pred], axis=1), tf.int32)

        tp_array, fn_array = tf.map_fn(
            lambda single_label_pred: self.get_label_detected(single_label_pred[0], single_label_pred[1]),
            y_true_pred,
            fn_output_signature=(tf.int32, tf.int32),
            parallel_iterations=cpu_count(),
        )
        tp = tf.reduce_sum(tp_array)
        fn = tf.reduce_sum(fn_array)
        return tp, fn

    @tf.function
    def get_label_detected(self, single_label, single_pred):
        indiced_label, indiced_pred = self._separate_predictions(single_label, single_pred)
        label_detected = tf.map_fn(
            lambda cancer_label: self.is_detected(cancer_label, indiced_pred),
            indiced_label,
            fn_output_signature=tf.TensorSpec(dtype=tf.bool, shape=[]),
            parallel_iterations=cpu_count(),
        )
        tp = tf.reduce_sum(tf.cast(label_detected, tf.int32))
        fn = tf.reduce_sum(tf.cast(~label_detected, tf.int32))
        return tp, fn

    @tf.function
    def is_detected(self, cancer_label, indiced_pred):
        IoUs = tf.map_fn(
            lambda pred: self._IoU(cancer_label, pred),
            indiced_pred,
            fn_output_signature=tf.TensorSpec(dtype=tf.float32, shape=[]),
            parallel_iterations=cpu_count()
        )
        return tf.reduce_any(IoUs > self.IoU_threshold)

    @tf.function
    def get_tp_fp(self, y_true, y_pred, sample_weight, threshold):
        if sample_weight is not None: raise NotImplementedError
        y_pred = tf.squeeze(tf.cast(y_pred > threshold, y_pred.dtype), -1)
        y_true_pred = tf.cast(tf.stack([y_true, y_pred], axis=1), tf.int32)

        tp_array, fp_array = tf.map_fn(
            lambda single_label_pred: self.get_tp_pred(single_label_pred[0], single_label_pred[1]),
            y_true_pred,
            fn_output_signature=(tf.int32, tf.int32),
            parallel_iterations=cpu_count(),
        )
        tp = tf.reduce_sum(tp_array)
        fp = tf.reduce_sum(fp_array)
        return tp, fp

    @tf.function
    def get_tp_pred(self, single_label, single_pred):
        indiced_label, indiced_pred = self._separate_predictions(single_label, single_pred)

        tp_pred = tf.map_fn(
            lambda cancer_pred: self.is_tp(cancer_pred, indiced_label),
            indiced_pred,
            fn_output_signature=tf.bool,
            parallel_iterations=cpu_count(),
        )
        tp = tf.reduce_sum(tf.cast(tp_pred, tf.int32))
        fp = tf.reduce_sum(tf.cast(~tp_pred, tf.int32))
        return tp, fp

    @tf.function
    def is_tp(self, cancer_pred, indiced_label):
        IoUs = tf.map_fn(
            lambda label: self._IoU(label, cancer_pred),
            indiced_label,
            fn_output_signature=tf.TensorSpec(dtype=tf.float32, shape=[]),
            parallel_iterations=cpu_count()
        )
        return tf.reduce_any(IoUs > self.IoU_threshold)

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
        tp_fn = tf.map_fn(
            lambda threshold: self.get_tp_fn(y_true, y_pred, sample_weight, threshold),
            self.thresholds,
            fn_output_signature=(tf.int32, tf.int32),
            parallel_iterations=cpu_count(),
        )
        self.tp_count.assign_add(tp_fn[0])
        self.fn_count.assign_add(tp_fn[1])
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
        tp_fp = tf.map_fn(
            lambda threshold: self.get_tp_fp(y_true, y_pred, sample_weight, threshold),
            self.thresholds,
            fn_output_signature=(tf.int32, tf.int32),
            parallel_iterations=cpu_count(),
        )
        self.tp_count.assign_add(tp_fp[0])
        self.fp_count.assign_add(tp_fp[1])
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
        tp_fn = tf.map_fn(
            lambda threshold: self.get_tp_fn(y_true, y_pred, sample_weight, threshold),
            self.thresholds,
            fn_output_signature=(tf.int32, tf.int32),
            parallel_iterations=cpu_count(),
        )
        self.tp_count.assign_add(tp_fn[0])
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
        tp_fp = tf.map_fn(
            lambda threshold: self.get_tp_fp(y_true, y_pred, sample_weight, threshold),
            self.thresholds,
            fn_output_signature=(tf.int32, tf.int32),
            parallel_iterations=cpu_count(),
        )
        self.fp_count.assign_add(tp_fp[1])
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
        tp_fn = tf.map_fn(
            lambda threshold: self.get_tp_fn(y_true, y_pred, sample_weight, threshold),
            self.thresholds,
            fn_output_signature=(tf.int32, tf.int32),
            parallel_iterations=cpu_count(),
        )
        self.fn_count.assign_add(tp_fn[1])
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
