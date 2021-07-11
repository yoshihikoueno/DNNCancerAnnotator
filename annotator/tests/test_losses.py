'''
unittests for loss related functions
'''

# built-in
import unittest
import random
from copy import deepcopy

# external
import tensorflow as tf

# custom
from annotator.utils import losses as custom_losses


class TestPositiveRate(unittest.TestCase):
    def setUp(self):
        self.endpoint = custom_losses.tf_get_positive_rate
        return

    def tearDown(self):
        return

    def prepare_label(self, positive_rate, ndims):
        '''Generate test data randomly

        Generated data `x` will satisfy the following conditions:
            1. x.shape[0] == `positive_rate`
            2. len(x.shape) == `ndims`

        Args:
            positive_rate: 1 dimensional array representing positive rates
            ndims: the num of dims for the generated data

        Returns:
            sample_label
        '''
        assert ndims > 0, 'ndims must be positive'
        slice_dimension = tf.random.uniform([ndims - 1], dtype=tf.int32, minval=8, maxval=20)
        sample_label = tf.map_fn(
            lambda rate: tf.cast(
                tf.random.uniform(slice_dimension, minval=0.001, maxval=0.999, dtype=tf.float32) > (1 - rate),
                tf.float32,
            ),
            positive_rate,
        )
        return sample_label

    def _run_test(self, positive_rate, ndims=5):
        test_data = self.prepare_label(positive_rate, ndims)
        positive_rate_out = self.endpoint(test_data)
        self.assertListEqual(positive_rate.shape.as_list(), positive_rate_out.shape.as_list())
        error = tf.reduce_sum(tf.abs(positive_rate_out - positive_rate))
        self.assertAlmostEqual(error.numpy(), 0, msg=f"{positive_rate.numpy()} vs {positive_rate_out.numpy()}", places=2)
        return

    def test_get_positive_rate_all_zero(self):
        self._run_test(tf.zeros([5]))
        return

    def test_get_positive_rate_all_one(self):
        self._run_test(tf.ones([5]))
        return

    def test_get_positive_rate_random(self):
        for _ in range(3):
            self._run_test(tf.random.uniform([4], minval=0, maxval=1, dtype=tf.float32), ndims=7)
        return
