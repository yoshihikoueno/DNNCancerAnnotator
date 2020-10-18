'''
provides various components which will help build DNN models easily
'''

# built-in
import pdb

# external
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from tensorflow.keras import Model


class Downsample(Layer):
    '''downsampling block'''
    def __init__(
        self,
        filters,
        rate,
        kernel_size,
        conv_stride,
        bn,
        n_conv=2,
        trainable=True,
        padding='valid',
        activation='relu',
        kernel_regularizer=None,
        **kargs,
    ):
        super().__init__(self, **kargs)
        self.configs = dict(
            filters=filters,
            rate=rate,
            kernel_size=kernel_size,
            conv_stride=conv_stride,
            bn=bn,
            n_conv=n_conv,
            trainable=trainable,
            padding=padding,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )
        self.padding = padding
        convs = [
            layers.Conv2D(
                filters, kernel_size, strides=conv_stride, padding=self.padding, activation=activation, trainable=trainable,
                kernel_regularizer=kernel_regularizer,
            )
            for i in range(n_conv)
        ]

        self.pool = layers.MaxPool2D([rate] * 2, strides=rate)

        if bn:
            self.batchnorms = [layers.BatchNormalization(trainable=trainable) for i in range(n_conv)]
            convs = [layer for tup in zip(convs, self.batchnorms) for layer in tup]
            self.pool = keras.Sequential([self.pool, layers.BatchNormalization(trainable=trainable)])

        self.convchain = keras.Sequential(convs)
        return

    def get_config(self):
        config = super().get_config()
        config.update(self.configs)
        return config

    def build(self, input_shape):
        self.convchain.build(input_shape)
        conv_output_shape = self.convchain.compute_output_shape(input_shape)
        self.pool.build(conv_output_shape)
        pool_output_shape = self.pool.compute_output_shape(conv_output_shape)
        self.built = True
        return conv_output_shape, pool_output_shape

    def call(self, inputs, training):
        conv = self.convchain(inputs, training=training)
        half = self.pool(conv, training=training)
        return conv, half


class Upsample(Layer):
    """upsampling block"""
    def __init__(
        self,
        filters,
        rate,
        kernel_size,
        conv_stride,
        bn,
        trainable,
        n_conv=2,
        padding='valid',
        activation='relu',
        kernel_regularizer=None,
        **kargs,
    ):
        super().__init__(self, **kargs)
        self.configs = dict(
            filters=filters,
            rate=rate,
            kernel_size=kernel_size,
            conv_stride=conv_stride,
            bn=bn,
            trainable=trainable,
            n_conv=n_conv,
            padding=padding,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            **kargs,
        )
        self.filters = filters
        self.rate = rate
        self.padding = padding
        self.activation = activation
        self.conv_transpose = layers.Convolution2DTranspose(
            filters=filters, kernel_size=rate, strides=rate, kernel_regularizer=kernel_regularizer,
            padding=self.padding, activation=None, trainable=trainable)

        self.conv_layers = [
            layers.Conv2D(
                filters=filters, kernel_size=kernel_size, kernel_regularizer=kernel_regularizer,
                strides=conv_stride, padding=self.padding, activation=self.activation, trainable=trainable
            ) for i in range(n_conv)
        ]

        if bn:
            bn_layers = [layers.BatchNormalization(trainable=trainable) for i in range(n_conv)]
            self.conv_transpose = keras.Sequential([self.conv_transpose, layers.BatchNormalization(trainable=trainable)])
            self.conv_layers = [layer for tup in zip(self.conv_layers, bn_layers) for layer in tup]

        self.convchain = keras.Sequential(self.conv_layers)
        return

    def get_config(self):
        config = super().get_config()
        config.update(self.configs)
        return config

    def build(self, input_shape, ref_shape):
        '''
            input_shapes: [ (inputs.shape), (reference.shape) ]
        '''
        self.conv_transpose.build(input_shape)
        conv_output_shape = self.conv_transpose.compute_output_shape(input_shape)
        convchain_input_shape = (*conv_output_shape[:3], + conv_output_shape[3] + ref_shape[3])
        self.convchain.build(convchain_input_shape)
        self.built = True
        return

    def compute_output_shape(self, input_shape, ref_shape):
        tconv_shape = self.conv_transpose.compute_output_shape(input_shape)
        output_shape = [*tconv_shape[:3], self.filters]
        return output_shape

    def call(self, inputs, reference, training):
        tconv0 = self.conv_transpose(inputs, training=training)
        assert all(map(lambda x, y: x >= y, reference.shape[1:], tconv0.shape[1:]))
        gap_half = (tf.shape(reference)[1:3] - tf.shape(tconv0)[1:3]) // 2
        cropped = tf.image.crop_to_bounding_box(reference, gap_half[0], gap_half[1], tf.shape(tconv0)[1], tf.shape(tconv0)[2])
        concatenated = tf.concat([tconv0, cropped], axis=-1)
        conved = self.convchain(concatenated, training=training)
        return conved


class Encoder(Layer):
    """encoder block"""
    def __init__(
        self,
        filters_first,
        n_downsample,
        rate,
        kernel_size,
        conv_stride,
        bn,
        trainable,
        n_conv=2,
        padding='valid',
        activation='relu',
        kernel_regularizer=None,
        **kargs,
    ):
        super().__init__(self, **kargs)
        self.configs = dict(
            filters_first=filters_first,
            n_downsample=n_downsample,
            rate=rate,
            kernel_size=kernel_size,
            conv_stride=conv_stride,
            bn=bn,
            trainable=trainable,
            n_conv=n_conv,
            padding=padding,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            **kargs,
        )
        self.padding = padding
        self.activation = activation
        self.downsamples = []
        next_filters = filters_first
        for i in range(n_downsample):
            self.downsamples.append(
                Downsample(
                    filters=next_filters,
                    rate=rate,
                    kernel_size=kernel_size,
                    conv_stride=conv_stride,
                    n_conv=n_conv,
                    bn=bn,
                    padding=self.padding,
                    trainable=trainable,
                    activation=self.activation,
                    kernel_regularizer=kernel_regularizer,
                )
            )
            next_filters = int(rate * next_filters)
        return

    def get_config(self):
        return self.configs

    def build(self, input_shape):
        ref_shapes = []
        output_shape = input_shape
        for downsample in self.downsamples:
            ref_shape, output_shape = downsample.build(output_shape)
            ref_shapes.append(ref_shape)
        self.built = True
        return output_shape, ref_shapes

    def call(self, inputs, training=False):
        res_list = list()
        next_inputs = inputs

        for downsample_layer in self.downsamples:
            res, downsampled = downsample_layer(
                inputs=next_inputs,
                training=training,
            )
            res_list.append(res)
            next_inputs = downsampled
        return res_list, downsampled


class Decoder(Layer):
    """decoder block"""
    def __init__(
        self,
        rate,
        kernel_size,
        conv_stride,
        bn,
        trainable,
        padding='valid',
        activation='relu',
        kernel_regularizer=None,
        **kargs,
    ):
        super().__init__(self, **kargs)
        self.configs = dict(
            rate=rate,
            kernel_size=kernel_size,
            conv_stride=conv_stride,
            bn=bn,
            trainable=trainable,
            padding=padding,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            **kargs,
        )
        self.upsamples = []
        self.rate = rate
        self.activation = activation
        self.kernel_size = kernel_size
        self.conv_stride = conv_stride
        self.bn = bn
        self.trainable = trainable
        self.padding = padding
        self.kernel_regularizer = kernel_regularizer
        return

    def get_config(self):
        config = super().get_config()
        config.update(self.configs)
        return config

    def build(self, inputs_shape, ref_shapes):
        for ref_shape in reversed(ref_shapes):
            self.upsamples.append(
                Upsample(
                    filters=ref_shape[-1],
                    rate=self.rate,
                    kernel_size=self.kernel_size,
                    conv_stride=self.conv_stride,
                    bn=self.bn,
                    trainable=self.trainable,
                    padding=self.padding,
                    activation=self.activation,
                    kernel_regularizer=self.kernel_regularizer,
                )
            )

        for upsample, ref_shape in zip(self.upsamples, reversed(ref_shapes)):
            upsample.build(inputs_shape, ref_shape)
            inputs_shape = upsample.compute_output_shape(inputs_shape, ref_shape)
        self.built = True
        return inputs_shape

    def call(self, inputs, res_list, training):
        upsampled = inputs
        assert len(res_list) == len(self.upsamples), f'#References {len(res_list)} != #upsamples {len(self.upsamples)}'
        for reference, upsample_layer in zip(reversed(res_list), self.upsamples):
            upsampled = upsample_layer(inputs=upsampled, reference=reference, training=training)
        return upsampled


def solve_activation(identifier):
    '''solve activation'''
    if callable(identifier):
        obj = identifier
    elif isinstance(identifier, str):
        obj = tf.keras.activations.get(identifier)
    elif isinstance(identifier, dict):
        obj = tf.keras.utils.deserialize_keras_object(
            identifier=identifier,
            module_objects=vars(tf.keras.layers)
        )
    else: raise ValueError(f'Failed to resolve activation: {identifier}')
    return obj
