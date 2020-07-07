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
        )
        self.padding = padding
        self.convs = [
            layers.Conv2D(
                filters, kernel_size, strides=conv_stride, padding=self.padding, activation=activation, trainable=trainable,
            )
            for i in range(n_conv)
        ]

        self.pool = layers.MaxPool2D([rate] * 2, strides=rate)

        if bn:
            self.batchnorms = [layers.BatchNormalization(trainable=trainable) for i in range(n_conv)]
            self.convs = [layer for tup in zip(self.convs, self.batchnorms) for layer in tup]
            self.pool = keras.Sequential([self.pool, layers.BatchNormalization(trainable=trainable)])

        self.convchain = keras.Sequential(self.convs)
        return

    def get_config(self):
        config = super().get_config()
        config.update(self.configs)
        return config

    def build(self, input_shape):
        conv_output_shape = self.convchain.compute_output_shape(input_shape)
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
        )
        self.filters = filters
        self.rate = rate
        self.padding = padding
        self.conv_transpose = layers.Convolution2DTranspose(
            filters=filters, kernel_size=rate, strides=rate,
            padding=self.padding, activation=None, trainable=trainable)

        self.conv_layers = [
            layers.Conv2D(
                filters=filters, kernel_size=kernel_size,
                strides=conv_stride, padding=self.padding, activation=tf.nn.relu, trainable=trainable
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
            **kargs,
        )
        self.padding = padding
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

    def call(self, inputs, training=True):
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
        )
        self.upsamples = []
        self.rate = rate
        self.kernel_size = kernel_size
        self.conv_stride = conv_stride
        self.bn = bn
        self.trainable = trainable
        self.padding = padding
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
