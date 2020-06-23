'''
provide various convinient functions regarding tf.data.Dataset
'''

# built-in
import os
import pdb
import sys

# external
import tensorflow as tf


def count(ds):
    size = 0
    for _ in ds: size += 1
    return size
