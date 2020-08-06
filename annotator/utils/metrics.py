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


class RegionBasedFBetaScore(tf.keras.metrics.Metric):
    def __init__(self, beta, thresholds, IoU_threshold=0.30, epsilon=1e-07, **kargs):
        super().__init__(**kargs)
        assert beta > 0
        self.beta = beta
        self.epsilon = epsilon
        self.thresholds = thresholds
        self.IoU_threshold = IoU_threshold
        self.precision = RegionBasedPrecision(threshold=self.thresholds, IoU_threshold=IoU_threshold)
        self.recall = RegionBasedRecall(threshold=self.thresholds, IoU_threshold=IoU_threshold)
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
        config.update({
            'beta': self.beta, 'epsilon': self.epsilon,
            'thresholds': self.thresholds, 'IoU_threshold': self.IoU_threshold,
        })
        return config


class RegionBasedRecall(tf.keras.metrics.Metric):
    def __init__(self, threshold, IoU_threshold=0.30, **kargs):
        super().__init__(**kargs)
        assert threshold > 0
        self.threshold = threshold
        self.IoU_threshold = IoU_threshold

        self.tp_count = self.add_weight(
            'tp_count', dtype=tf.int32, initializer=tf.zeros_initializer)
        self.fn_count = self.add_weight(
            'fn_count', dtype=tf.int32, initializer=tf.zeros_initializer)
        return

    def update_state(self, y_true, y_pred, sample_weight=None):
        if sample_weight is not None: raise NotImplementedError
        y_pred = tf.cast(y_pred > self.threshold, y_pred.dtype)
        y_true_pred = tf.cast(tf.stack([y_true, y_pred], axis=1), tf.int32)

        for single_label, single_pred in y_true_pred:
            indiced_label = tf.one_hot(tfa.image.connected_components(single_label), tf.reduce_max(single_label) + 1)[:, :, 1:]
            indiced_label = tf.cast(tf.transpose(indiced_label, [2, 0, 1]), tf.bool)
            indiced_pred = tf.one_hot(tfa.image.connected_components(single_pred), tf.reduce_max(single_pred) + 1)[:, :, 1:]
            indiced_pred = tf.cast(tf.transpose(indiced_pred, [2, 0, 1]), tf.bool)
            for cancer_label in indiced_label:
                IoUs = tf.map_fn(
                    lambda pred: self._IoU(cancer_label, pred),
                    indiced_pred,
                    dtype=tf.float32,
                    parallel_iterations=cpu_count()
                )
                if tf.reduce_any(IoUs > self.IoU_threshold): self.tp_count.assign_add(1)
                else: self.fn_count.assign_add(1)
        return

    @tf.function
    def _IoU(self, cancer_label, cancer_pred):
        intersection = tf.reduce_sum(tf.cast(cancer_label & cancer_pred, tf.float32))
        union = tf.reduce_sum(tf.cast(cancer_label | cancer_pred, tf.float32))
        iou = intersection / union
        return iou

    def result(self):
        result = self.tp_count / (self.tp_count + self.fn_count)
        return result

    def get_config(self):
        configs = super().get_config()
        configs['threshold'] = self.threshold
        configs['IoU_threshold'] = self.IoU_threshold
        return configs


class RegionBasedPrecision(tf.keras.metrics.Metric):
    def __init__(self, threshold, IoU_threshold=0.30, **kargs):
        super().__init__(**kargs)
        assert threshold > 0
        self.threshold = threshold
        self.IoU_threshold = IoU_threshold

        self.tp_count = self.add_weight(
            'tp_count', dtype=tf.int32, initializer=tf.zeros_initializer)
        self.fp_count = self.add_weight(
            'fp_count', dtype=tf.int32, initializer=tf.zeros_initializer)
        return

    def update_state(self, y_true, y_pred, sample_weight=None):
        if sample_weight is not None: raise NotImplementedError
        y_pred = tf.cast(y_pred > self.threshold, y_pred.dtype)
        y_true_pred = tf.cast(tf.stack([y_true, y_pred], axis=1), tf.int32)

        for single_label, single_pred in y_true_pred:
            indiced_label = tf.one_hot(tfa.image.connected_components(single_label), tf.reduce_max(single_label) + 1)[:, :, 1:]
            indiced_label = tf.cast(tf.transpose(indiced_label, [2, 0, 1]), tf.bool)
            indiced_pred = tf.one_hot(tfa.image.connected_components(single_pred), tf.reduce_max(single_pred) + 1)[:, :, 1:]
            indiced_pred = tf.cast(tf.transpose(indiced_pred, [2, 0, 1]), tf.bool)
            for cancer_pred in indiced_pred:
                IoUs = tf.map_fn(
                    lambda label: self._IoU(label, cancer_pred),
                    indiced_label,
                    dtype=tf.float32,
                    parallel_iterations=cpu_count()
                )
                if tf.reduce_any(IoUs > self.IoU_threshold): self.tp_count.assign_add(1)
                else: self.fp_count.assign_add(1)
        return

    @tf.function
    def _IoU(self, cancer_label, cancer_pred):
        intersection = tf.reduce_sum(tf.cast(cancer_label & cancer_pred, tf.float32))
        union = tf.reduce_sum(tf.cast(cancer_label | cancer_pred, tf.float32))
        iou = intersection / union
        return iou

    def result(self):
        result = self.tp_count / (self.tp_count + self.fp_count)
        return result

    def get_config(self):
        configs = super().get_config()
        configs['threshold'] = self.threshold
        configs['IoU_threshold'] = self.IoU_threshold
        return configs


tf.keras.utils.get_custom_objects().update(FBetaScore=FBetaScore)
tf.keras.utils.get_custom_objects().update(RegionBasedRecall=RegionBasedRecall)
