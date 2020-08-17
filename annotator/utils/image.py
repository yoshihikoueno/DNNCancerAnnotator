'''
provide various functions for image manipulation
'''

# built-in
import pdb

# external
import tensorflow as tf


def morph_open(image, filter_size):
    '''apply morphological opening op

    Args:
        image: 4D image
            shape: batch, H, W, C
        filter_size: size of kernel filter
    '''
    filter_ = tf.zeros([filter_size, filter_size, 1], dtype=image.dtype)
    strides = [1] * 4
    dilations = [1, 1, 1, 1]

    eroded = tf.nn.erosion2d(image, filter_, strides, 'SAME', 'NHWC', dilations)
    dilated = tf.nn.dilation2d(eroded, filter_, strides, 'SAME', 'NHWC', dilations)
    return dilated
