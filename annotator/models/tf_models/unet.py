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
from annotator.models.tf_models import components


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
        **kargs,
    ):
        super().__init__(**kargs)
        self.encoder = components.Encoder(
            filters_first=filters_first,
            n_downsample=n_downsample,
            rate=rate,
            kernel_size=kernel_size,
            conv_stride=conv_stride,
            bn=bn,
            padding=padding,
            trainable=trainable,
        )
        self.decoder = components.Decoder(
            rate=rate,
            kernel_size=kernel_size,
            conv_stride=conv_stride,
            bn=bn,
            padding=padding,
            trainable=trainable,
        )
        return

    def build(self, input_shape):
        self.encoder_output_shape, self.ref_shapes = self.encoder.build(input_shape)
        decoder_out = self.decoder.build(self.encoder_output_shape, self.ref_shapes)
        self.built = True
        return decoder_out

    def __call__(self, inputs, training=True):
        res_list, downsampled = self.encoder(inputs=inputs, training=training)
        output = self.decoder(inputs=downsampled, res_list=res_list, training=training)
        return output


class UNetAnnotator(keras.Sequential):
    def __init__(
        self,
        n_filters_first,
        n_downsample,
        rate,
        kernel_size,
        conv_stride,
        bn=False,
        padding='valid',
        **kargs,
    ):
        '''
        this function represents a part of model which will produce annotation or segmentation
        Args:
            input_: input tensor
            n_filters_first: the num of filters in the first block
            n_downsample: the num of downsample
            rate: the rate of downsample and upsample
            kernel_size: kernel_size of every Conv
            conv_stride: stride in conv
            bn: (bool) whether or not BN is applied
            training: (bool) whether the model is being trained
                this can be None as long as bn=False
            padding: padding method used in internal components
            trainable: (bool) whether or not this block is trainable
        '''
        unet = UNet(
            filters_first=n_filters_first,
            n_downsample=n_downsample,
            rate=rate,
            kernel_size=kernel_size,
            conv_stride=conv_stride,
            bn=bn,
            padding=padding,
            **kargs,
        )
        last_conv = layers.Conv2D(filters=1, kernel_size=1, padding=padding, **kargs)
        super().__init__([unet, last_conv], **kargs)
        self.unet = unet
        self.padding = padding
        self.last_conv = last_conv
        return

    def build(self, input_shape):
        unet_out = self.unet.build(input_shape)
        self.last_conv.build(unet_out)
        # self.built = True
        super().build(input_shape)
        return
