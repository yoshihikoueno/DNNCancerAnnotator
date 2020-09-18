'''
unittests for region based metrics
'''

# built-in
import unittest
import pdb
import random
from copy import deepcopy

# external
import tensorflow as tf

# custom
from annotator.utils import metrics as custom_metrics


class TestRegionMetricsSingleThreshold(unittest.TestCase):
    def setUp(self):
        self.metric = custom_metrics.RegionBasedConfusionMatrix(
            thresholds=0.5,
            IoU_threshold=0.3,
            resize_factor=1.0,
        )
        self.batch_size = 10
        self.radius = tf.random.uniform([self.batch_size], 10, 30, tf.int64)
        self.center_x = tf.random.uniform([self.batch_size], 30, 70, tf.int32)
        self.center_y = tf.random.uniform([self.batch_size], 80, 120, tf.int32)
        self.center_x_off = tf.random.uniform([self.batch_size], 130, 170, tf.int32)
        self.center_y_off = tf.random.uniform([self.batch_size], 80, 120, tf.int32)
        self.width = 200
        self.height = 200
        self.n_threshold = 1
        return

    def tearDown(self):
        self.metric = None
        tf.keras.backend.clear_session()
        return

    def test_tp_fn_all_tp(self):
        y_true, y_pred, n_tp, n_fn = self.generate_tp_fn_samples(1.0)
        tp, fn = self.metric.get_tp_fn(y_true, y_pred, None)
        self.assertListEqual(tp.numpy().tolist(), [n_tp] * self.n_threshold, f'tp: {tp}, fn: {fn}')
        self.assertListEqual(fn.numpy().tolist(), [n_fn] * self.n_threshold, f'tp: {tp}, fn: {fn}')
        return

    def test_tp_fn_all_fn(self):
        y_true, y_pred, n_tp, n_fn = self.generate_tp_fn_samples(0.0)
        tp, fn = self.metric.get_tp_fn(y_true, y_pred, None)
        self.assertListEqual(tp.numpy().tolist(), [n_tp] * self.n_threshold, f'tp: {tp}, fn: {fn}')
        self.assertListEqual(fn.numpy().tolist(), [n_fn] * self.n_threshold, f'tp: {tp}, fn: {fn}')
        return

    def test_tp_fn_all_fp(self):
        y_true, y_pred, n_tp, n_fp = self.generate_tp_fp_samples(0.0)
        tp, fn = self.metric.get_tp_fn(y_true, y_pred, None)
        self.assertListEqual(tp.numpy().tolist(), [0] * self.n_threshold, f'tp: {tp}, fn: {fn}')
        self.assertListEqual(fn.numpy().tolist(), [0] * self.n_threshold, f'tp: {tp}, fn: {fn}')
        return

    def test_tp_fn_half(self):
        y_true, y_pred, n_tp, n_fn = self.generate_tp_fn_samples(0.5)
        tp, fn = self.metric.get_tp_fn(y_true, y_pred, None)
        self.assertListEqual(tp.numpy().tolist(), [n_tp] * self.n_threshold, f'tp: {tp}, fn: {fn}')
        self.assertListEqual(fn.numpy().tolist(), [n_fn] * self.n_threshold, f'tp: {tp}, fn: {fn}')
        return

    def test_tp_fp_all_tp(self):
        y_true, y_pred, n_tp, n_fp = self.generate_tp_fp_samples(0.0)
        tp, fp = self.metric.get_tp_fp(y_true, y_pred, None)
        self.assertListEqual(tp.numpy().tolist(), [n_tp] * self.n_threshold, f'tp: {tp}, fp: {fp}')
        self.assertListEqual(fp.numpy().tolist(), [n_fp] * self.n_threshold, f'tp: {tp}, fp: {fp}')
        return

    def test_tp_fp_all_fp(self):
        y_true, y_pred, n_tp, n_fp = self.generate_tp_fp_samples(1.0)
        tp, fp = self.metric.get_tp_fp(y_true, y_pred, None)
        self.assertListEqual(tp.numpy().tolist(), [n_tp] * self.n_threshold, f'tp: {tp}, fp: {fp}')
        self.assertListEqual(fp.numpy().tolist(), [n_fp] * self.n_threshold, f'tp: {tp}, fp: {fp}')
        return

    def test_tp_fp_all_fn(self):
        y_true, y_pred, n_tp, n_fn = self.generate_tp_fn_samples(0.0)
        tp, fp = self.metric.get_tp_fp(y_true, y_pred, None)
        self.assertListEqual(tp.numpy().tolist(), [0] * self.n_threshold, f'tp: {tp}, fp: {fp}')
        self.assertListEqual(fp.numpy().tolist(), [0] * self.n_threshold, f'tp: {tp}, fp: {fp}')
        return

    def test_tp_fp_half(self):
        y_true, y_pred, n_tp, n_fp = self.generate_tp_fp_samples(0.5)
        tp, fp = self.metric.get_tp_fp(y_true, y_pred, None)
        self.assertListEqual(tp.numpy().tolist(), [n_tp] * self.n_threshold, f'tp: {tp}, fp: {fp}')
        self.assertListEqual(fp.numpy().tolist(), [n_fp] * self.n_threshold, f'tp: {tp}, fp: {fp}')
        return

    def test_tp_fn_fp_all_tp(self):
        y_true, y_pred, n_tp, n_fp = self.generate_tp_fp_samples(0.0)
        tp, fn, fp = self.metric.get_tp_fn_fp(y_true, y_pred, None)
        self.assertListEqual(tp.numpy().tolist(), [n_tp] * self.n_threshold)
        self.assertListEqual(fn.numpy().tolist(), [0] * self.n_threshold)
        self.assertListEqual(fp.numpy().tolist(), [n_fp] * self.n_threshold)
        return

    def test_tp_fn_fp_all_fp(self):
        y_true, y_pred, n_tp, n_fp = self.generate_tp_fp_samples(1.0)
        tp, fn, fp = self.metric.get_tp_fn_fp(y_true, y_pred, None)
        self.assertListEqual(tp.numpy().tolist(), [n_tp] * self.n_threshold)
        self.assertListEqual(fn.numpy().tolist(), [0] * self.n_threshold)
        self.assertListEqual(fp.numpy().tolist(), [n_fp] * self.n_threshold)
        return

    def test_tp_fn_fp_all_fn(self):
        y_true, y_pred, n_tp, n_fn = self.generate_tp_fn_samples(0.0)
        tp, fn, fp = self.metric.get_tp_fn_fp(y_true, y_pred, None)
        self.assertListEqual(tp.numpy().tolist(), [0] * self.n_threshold)
        self.assertListEqual(fn.numpy().tolist(), [n_fn] * self.n_threshold)
        self.assertListEqual(fp.numpy().tolist(), [0] * self.n_threshold)
        return

    def test_tp_fn_fp_half(self):
        y_true, y_pred, n_tp, n_fp = self.generate_tp_fp_samples(0.5)
        tp, fn, fp = self.metric.get_tp_fn_fp(y_true, y_pred, None)
        self.assertListEqual(tp.numpy().tolist(), [n_tp] * self.n_threshold)
        self.assertListEqual(fn.numpy().tolist(), [0] * self.n_threshold)
        self.assertListEqual(fp.numpy().tolist(), [n_fp] * self.n_threshold)
        return

    def test_tp_fn_fp_null(self):
        y_true, y_pred = self.generate_null_samples()
        tp, fn, fp = self.metric.get_tp_fn_fp(y_true, y_pred, None)
        self.assertListEqual(tp.numpy().tolist(), [0] * self.n_threshold)
        self.assertListEqual(fn.numpy().tolist(), [0] * self.n_threshold)
        self.assertListEqual(fp.numpy().tolist(), [0] * self.n_threshold)
        return

    def test_tp_fn_fp_mixed(self):
        y_true, y_pred, n_tp, n_fn = self.generate_tp_fn_samples(0.4)
        offs, n_off = self.generate_off_samples(0.7)
        y_pred = y_pred + offs
        n_fp = n_off
        tp, fn, fp = self.metric.get_tp_fn_fp(y_true, y_pred, None)
        self.assertListEqual(tp.numpy().tolist(), [n_tp] * self.n_threshold)
        self.assertListEqual(fn.numpy().tolist(), [n_fn] * self.n_threshold)
        self.assertListEqual(fp.numpy().tolist(), [n_fp] * self.n_threshold)
        return

    def test_consistency(self):
        y_true, y_pred, n_tp, n_fn = self.generate_tp_fn_samples(0.4)
        offs, n_off = self.generate_off_samples(0.7)
        y_pred = y_pred + offs
        tp, fn, fp = self.metric.get_tp_fn_fp(y_true, y_pred, None)
        tp2, fn2 = self.metric.get_tp_fn(y_true, y_pred, None)
        _, fp2 = self.metric.get_tp_fp(y_true, y_pred, None)
        self.assertListEqual(tp.numpy().tolist(), tp2.numpy().tolist())
        self.assertListEqual(fp.numpy().tolist(), fp2.numpy().tolist())
        self.assertListEqual(fn.numpy().tolist(), fn2.numpy().tolist())
        return

    def test_consistency_random(self):
        for i in range(10):
            y_true, y_pred = self.generate_random_samples(20)

            tp, fn, fp = self.metric.get_tp_fn_fp(y_true, y_pred, None)
            tp2, fn2 = self.metric.get_tp_fn(y_true, y_pred, None)
            _, fp2 = self.metric.get_tp_fp(y_true, y_pred, None)

            self.assertListEqual(tp.numpy().tolist(), tp2.numpy().tolist())
            self.assertListEqual(fn.numpy().tolist(), fn2.numpy().tolist())
            self.assertListEqual(fp.numpy().tolist(), fp2.numpy().tolist())
        return

    def test_highlevel_consistency(self):
        tp_count = custom_metrics.RegionBasedTruePositives(**self.metric.get_config())
        fp_count = custom_metrics.RegionBasedFalsePositives(**self.metric.get_config())
        fn_count = custom_metrics.RegionBasedFalseNegatives(**self.metric.get_config())
        precision_count = custom_metrics.RegionBasedPrecision(**self.metric.get_config())
        recall_count = custom_metrics.RegionBasedRecall(**self.metric.get_config())
        confusion_count = custom_metrics.RegionBasedConfusionMatrix(**self.metric.get_config())

        for i in range(5):
            y_true, y_pred = self.generate_random_samples(20)
            tp_count.update_state(y_true, y_pred)
            fp_count.update_state(y_true, y_pred)
            fn_count.update_state(y_true, y_pred)
            precision_count.update_state(y_true, y_pred)
            recall_count.update_state(y_true, y_pred)
            confusion_count.update_state(y_true, y_pred)

        if self.n_threshold == 1:
            self.assertEqual(
                tp_count.result().numpy().tolist(),
                confusion_count.result_dict()['true_positive_counts'].numpy().tolist(),
            )
            self.assertEqual(
                fp_count.result().numpy().tolist(),
                confusion_count.result_dict()['false_positive_counts'].numpy().tolist(),
            )
            self.assertEqual(
                fn_count.result().numpy().tolist(),
                confusion_count.result_dict()['false_negative_counts'].numpy().tolist(),
            )
            self.assertEqual(
                precision_count.result().numpy().tolist(),
                confusion_count.result_dict()['precision'].numpy().tolist(),
            )
            self.assertEqual(
                recall_count.result().numpy().tolist(),
                confusion_count.result_dict()['recall'].numpy().tolist(),
            )
        else:
            self.assertListEqual(
                tp_count.result().numpy().tolist(),
                confusion_count.result_dict()['true_positive_counts'].numpy().tolist(),
            )
            self.assertListEqual(
                fp_count.result().numpy().tolist(),
                confusion_count.result_dict()['false_positive_counts'].numpy().tolist(),
            )
            self.assertListEqual(
                fn_count.result().numpy().tolist(),
                confusion_count.result_dict()['false_negative_counts'].numpy().tolist(),
            )
            self.assertListEqual(
                precision_count.result().numpy().tolist(),
                confusion_count.result_dict()['precision'].numpy().tolist(),
            )
            self.assertListEqual(
                recall_count.result().numpy().tolist(),
                confusion_count.result_dict()['recall'].numpy().tolist(),
            )
        return

    def generate_tp_fn_samples(self, tp_rate):
        y_true = tf.stack(
            [
                self.draw_circle(
                    tf.zeros([self.width, self.height], tf.int64),
                    r, cx, cy
                ) for r, cx, cy in zip(self.radius, self.center_x, self.center_y)
            ],
            axis=0,
        )
        y_pred = tf.expand_dims(tf.cast(y_true, tf.float32), -1)
        n_tp = int(self.batch_size * tp_rate)
        n_fn = self.batch_size - n_tp
        tp_indicator = tf.random.shuffle(tf.concat([tf.ones([n_tp], y_pred.dtype), tf.zeros([n_fn], y_pred.dtype)], -1))
        y_pred = tf.transpose(tf.transpose(y_pred) * tp_indicator)
        return y_true, y_pred, n_tp, n_fn

    def generate_random_samples(self, nslices, min_=1.0, max_=1.0):
        def gen_slice(dtype, ncircles, min_=1.0, max_=1.0):
            image = tf.zeros([self.width, self.height], dtype)
            for _ in range(ncircles):
                image = self.draw_circle(
                    image,
                    random.uniform(5.0, self.width / 20),
                    random.uniform(0.0, self.width),
                    random.uniform(0.0, self.height),
                    min_,
                    max_,
                )
            return image

        y_true = tf.stack(
            [gen_slice(tf.int32, 5) for _ in range(nslices)],
            axis=0,
        )

        y_pred = tf.stack(
            [gen_slice(tf.float32, 5, min_, max_) for _ in range(nslices)],
            axis=0,
        )

        y_pred = tf.expand_dims(y_pred, -1)
        return y_true, y_pred

    def generate_null_samples(self):
        y_true = tf.zeros([self.batch_size, self.width, self.height], tf.int64)
        y_pred = tf.expand_dims(tf.cast(y_true, tf.float32), -1)
        return y_true, y_pred

    def generate_tp_fp_samples(self, tp_rate):
        y_true = tf.stack(
            [
                self.draw_circle(
                    tf.zeros([self.width, self.height], tf.int64),
                    r, cx, cy
                ) for r, cx, cy in zip(self.radius, self.center_x, self.center_y)
            ],
            axis=0,
        )
        y_pred = tf.expand_dims(tf.cast(y_true, tf.float32), -1)
        n_tp = int(self.batch_size * tp_rate)
        n_fp = self.batch_size - n_tp
        tp_indicator = tf.random.shuffle(tf.concat([tf.ones([n_tp], y_true.dtype), tf.zeros([n_fp], y_true.dtype)], -1))
        y_true = tf.transpose(tf.transpose(y_true) * tp_indicator)
        return y_true, y_pred, n_tp, n_fp

    def generate_off_samples(self, off_rate):
        offs = tf.stack(
            [
                self.draw_circle(
                    tf.zeros([self.width, self.height], tf.int64),
                    r, cx, cy
                ) for r, cx, cy in zip(self.radius, self.center_x_off, self.center_y_off)
            ],
            axis=0,
        )
        offs = tf.expand_dims(tf.cast(offs, tf.float32), -1)
        n_off = int(self.batch_size * off_rate)
        off_indicator = tf.random.shuffle(
            tf.concat([tf.ones([n_off], offs.dtype), tf.zeros([self.batch_size - n_off], offs.dtype)], -1)
        )
        offs = tf.transpose(tf.transpose(offs) * off_indicator)
        return offs, n_off

    @classmethod
    def draw_circle(cls, tensor, radius, center_x, center_y, min_=1.0, max_=1.0):
        assert len(tensor.shape) == 2
        width, height = tensor.shape
        center_x, center_y = tf.cast(center_x, tensor.dtype), tf.cast(center_y, tensor.dtype)

        x_indices = tf.range(0, width, dtype=tensor.dtype)
        y_indices = tf.range(0, height, dtype=tensor.dtype)

        x_dist = (x_indices - center_x)**2
        x_dist = tf.broadcast_to(x_dist, [x_indices.shape[0], x_indices.shape[0]])
        y_dist = (y_indices - center_y)**2
        y_dist = tf.transpose(tf.broadcast_to(y_dist, [y_indices.shape[0], y_indices.shape[0]]))

        dist = tf.sqrt(tf.cast(x_dist + y_dist, tf.float32))
        output = tf.cast(dist < tf.cast(radius, dist.dtype), tensor.dtype)
        output = tf.cast(tf.cast(output, tf.float32) * random.uniform(min_, max_), tensor.dtype)
        output = output + tensor
        return output


class TestRegionMetricsMultiThreshold(TestRegionMetricsSingleThreshold):
    def setUp(self):
        super().setUp()
        self.n_threshold = 10
        configs = self.metric.get_config()
        configs['thresholds'] = [i / (self.n_threshold - 1) for i in range(self.n_threshold)]
        self.metric = custom_metrics.RegionBasedConfusionMatrix(**configs)
        return

    def test_consistency_multithresholds(self):
        thresholds = self.metric.thresholds
        config = self.metric.get_config()

        def new_config(threshold, config):
            config = deepcopy(config)
            config['thresholds'] = [threshold]
            return config

        confusion_count = custom_metrics.RegionBasedConfusionMatrix(**config)
        confusion_count_list = [
            custom_metrics.RegionBasedConfusionMatrix(**new_config(threshold, config))
            for threshold in thresholds
        ]

        y_true, y_pred = self.generate_random_samples(20, 0.2, 1.0)
        tp, fn, fp = confusion_count.get_tp_fn_fp(y_true, y_pred, None)

        tp_fn_fp_list = [m.get_tp_fn_fp(y_true, y_pred, None) for m in confusion_count_list]
        tp_list = list(map(lambda x: x[0].numpy().tolist()[0], tp_fn_fp_list))
        fn_list = list(map(lambda x: x[1].numpy().tolist()[0], tp_fn_fp_list))
        fp_list = list(map(lambda x: x[2].numpy().tolist()[0], tp_fn_fp_list))

        self.assertListEqual(tp_list, tp.numpy().tolist())
        self.assertListEqual(fn_list, fn.numpy().tolist())
        self.assertListEqual(fp_list, fp.numpy().tolist())
        return

    def test_highlevel_consistency_multithresholds(self):
        thresholds = self.metric.thresholds
        config = self.metric.get_config()

        def new_config(threshold, config):
            config = deepcopy(config)
            config['thresholds'] = [threshold]
            return config

        confusion_count = custom_metrics.RegionBasedConfusionMatrix(**config)
        confusion_count_list = [
            custom_metrics.RegionBasedConfusionMatrix(**new_config(threshold, config))
            for threshold in thresholds
        ]

        for i in range(10):
            y_true, y_pred = self.generate_random_samples(20, 0.2, 1.0)
            confusion_count.update_state(y_true, y_pred)
            for m in confusion_count_list: m.update_state(y_true, y_pred)

        self.assertListEqual(
            [m.result_dict()['true_positive_counts'].numpy().tolist() for m in confusion_count_list],
            confusion_count.result_dict()['true_positive_counts'].numpy().tolist(),
        )
        self.assertListEqual(
            [m.result_dict()['false_positive_counts'].numpy().tolist() for m in confusion_count_list],
            confusion_count.result_dict()['false_positive_counts'].numpy().tolist(),
        )
        self.assertListEqual(
            [m.result_dict()['false_negative_counts'].numpy().tolist() for m in confusion_count_list],
            confusion_count.result_dict()['false_negative_counts'].numpy().tolist(),
        )
        self.assertListEqual(
            [m.result_dict()['precision_counts'].numpy().tolist() for m in confusion_count_list],
            confusion_count.result_dict()['precision_counts'].numpy().tolist(),
        )
        self.assertListEqual(
            [m.result_dict()['recall_counts'].numpy().tolist() for m in confusion_count_list],
            confusion_count.result_dict()['recall_counts'].numpy().tolist(),
        )
        return

class TestRegionMetricsSingleThresholdShrinked(TestRegionMetricsSingleThreshold):
    def setUp(self):
        super().setUp()
        configs = self.metric.get_config()
        configs['resize_factor'] = 0.5
        self.metric = custom_metrics.RegionBasedConfusionMatrix(**configs)
        return

class TestRegionMetricsMultiThresholdShrinked(TestRegionMetricsMultiThreshold):
    def setUp(self):
        super().setUp()
        configs = self.metric.get_config()
        configs['resize_factor'] = 0.5
        self.metric = custom_metrics.RegionBasedConfusionMatrix(**configs)
        return
