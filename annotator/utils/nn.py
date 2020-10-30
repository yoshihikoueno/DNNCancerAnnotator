'''
module to provide various tensor manipulation funcitons
'''

# built-in
import pdb

# external
import tensorflow as tf


def to_distributed(data, axis=0):
    '''return a func which distributes values'''
    def func(ctx):
        len_ = tf.shape(data)[axis]
        num_elems = len_ // ctx.num_replicas_in_sync
        start_idx = ctx.replica_id_in_sync_group * num_elems
        if ctx.replica_id_in_sync_group + 1 == ctx.num_replicas_in_sync:
            num_elems += len_ % ctx.num_replicas_in_sync
        indices = tf.range(start_idx, start_idx + num_elems)
        output = tf.gather(data, indices, axis=axis)
        return output
    return tf.distribute.get_strategy().experimental_distribute_values_from_function(func)
