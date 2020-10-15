'''
UNet
'''

# built-in
import pdb

# external
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from tensorflow.keras import Model

# customs
from . import components


class UNet(Layer):
    '''U-Net'''
    def __init__(
        self,
        filters_first,
        n_downsample,
        rate,
        kernel_size,
        conv_stride,
        bn=False,
        trainable=True,
        padding='valid',
        activation='relu',
        **kargs,
    ):
        super().__init__(**kargs)
        self.configs = dict(
            filters_first=filters_first,
            n_downsample=n_downsample,
            rate=rate,
            kernel_size=kernel_size,
            conv_stride=conv_stride,
            bn=bn,
            trainable=trainable,
            padding=padding,
            activation=activation,
            **kargs,
        )
        self.encoder = components.Encoder(
            filters_first=filters_first,
            n_downsample=n_downsample,
            rate=rate,
            kernel_size=kernel_size,
            conv_stride=conv_stride,
            bn=bn,
            padding=padding,
            activation=activation,
            trainable=trainable,
        )
        self.decoder = components.Decoder(
            rate=rate,
            kernel_size=kernel_size,
            conv_stride=conv_stride,
            bn=bn,
            padding=padding,
            activation=activation,
            trainable=trainable,
        )
        return

    def get_config(self):
        config = super().get_config()
        config.update(self.configs)
        return config

    def build(self, input_shape):
        self.encoder_output_shape, self.ref_shapes = self.encoder.build(input_shape)
        decoder_out = self.decoder.build(self.encoder_output_shape, self.ref_shapes)
        self.built = True
        return decoder_out

    def call(self, inputs, training=True):
        res_list, downsampled = self.encoder(inputs=inputs, training=training)
        output = self.decoder(inputs=downsampled, res_list=res_list, training=training)
        return output


class UNetAnnotator(keras.Model):
    def __init__(
        self,
        n_filters_first,
        n_downsample,
        rate,
        kernel_size,
        conv_stride,
        bn=False,
        padding='valid',
        activation='relu',
        **kargs,
    ):
        '''
        A class that represents a part of model which will produce annotation or segmentation

        Args:
            input_: input tensor
            n_filters_first: the num of filters in the first block
            n_downsample: the num of downsample
            rate: the rate of downsample and upsample
            kernel_size: kernel_size of every Conv
            conv_stride: stride in conv
            bn (bool): whether or not BN is applied
            training (bool): whether the model is being trained
                this can be None as long as bn=False
            padding: padding method used in internal components
            trainable (bool): whether or not this block is trainable
        '''
        super().__init__(**kargs)
        self.configs = dict(
            n_filters_first=n_filters_first,
            n_downsample=n_downsample,
            rate=rate,
            kernel_size=kernel_size,
            conv_stride=conv_stride,
            bn=bn,
            padding=padding,
            activation=activation,
            **kargs,
        )
        unet = UNet(
            filters_first=n_filters_first,
            n_downsample=n_downsample,
            rate=rate,
            kernel_size=kernel_size,
            conv_stride=conv_stride,
            bn=bn,
            padding=padding,
            activation=components.solve_activation(activation),
            **kargs,
        )
        last_conv = layers.Conv2D(filters=1, kernel_size=1, activation='sigmoid', padding=padding, **kargs)
        self.unet = unet
        self.padding = padding
        self.last_conv = last_conv
        return

    def get_config(self):
        return self.configs

    @classmethod
    def from_config(cls, config):
        instance = cls(**config)
        return instance

    def build(self, input_shape):
        unet_out = self.unet.build(input_shape)
        self.last_conv.build(unet_out)
        self.built = True
        return

    @tf.function
    def call(self, x, training=True):
        unet_out = self.unet(x, training=training)
        output = self.last_conv(unet_out, training=training)
        return output
