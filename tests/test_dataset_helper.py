import unittest
import numpy as np
import dataset_helpers.prostate_cancer_utils as prostate_cancer_utils
import pdb
import tensorflow as tf
tf.enable_eager_execution()


class TestProstateCancerUtils(unittest.TestCase):
    def test_decode_mri(self):
        image_path = 'tests/resources/image_gray.jpg'
        target_nchannels = 1
        from_new = prostate_cancer_utils.decode_mri(image_path, target_nchannels=target_nchannels).numpy()
        from_old = tf.cast(tf.image.decode_jpeg(tf.io.read_file(image_path), channels=target_nchannels), tf.float32).numpy()
        self.assertTrue(np.all(from_new == from_old))
        return


if __name__ == '__main__':
    unittest.main()
