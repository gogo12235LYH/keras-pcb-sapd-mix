import tensorflow as tf
from tensorflow_addons.layers import GroupNormalization
from models.layers import WSConv2D, AlignLayer
import numpy as np
import tensorflow.keras as keras
import config

_kernel_initializer = tf.initializers.RandomNormal(0.0, 0.01)


class AlignHead(keras.Model):
    def __init__(self, width=256, depth=4, num_cls=20, ws=True, *args, **kwargs):
        super(AlignHead, self).__init__(name='AlignHead', *args, **kwargs)
        self.width = width

        _conv2d_setting = {
            'filters': width,
            'kernel_size': 3,
            'strides': 1,
            'padding': 'same',
            'kernel_initializer': _kernel_initializer
        }

        self.blocks_a = keras.Sequential()
        for _ in range(depth):
            # conv2d
            self.blocks_a.add(keras.layers.Conv2D(**_conv2d_setting))
            # activation
            self.blocks_a.add(keras.layers.Activation(tf.nn.relu))

        self.blocks_b = keras.Sequential()
        for _ in range(depth):
            # weight standardization
            if ws:
                # conv2d with WS
                self.blocks_b.add(WSConv2D(**_conv2d_setting))
            else:
                # conv2d
                self.blocks_b.add(keras.layers.Conv2D(**_conv2d_setting))

            # layer normalization (g16)
            self.blocks_b.add(GroupNormalization(groups=config.HEAD_GROUPS, epsilon=1e-5))
            # activation
            self.blocks_b.add(keras.layers.Activation(tf.nn.relu))

        # Classification and Regression convolution output
        self.cls_conv2d = keras.layers.Conv2D(
            filters=num_cls, kernel_size=3, strides=1, padding='same',
            kernel_initializer=_kernel_initializer,
            bias_initializer=tf.constant_initializer(-np.log((1 - 0.01) / 0.01)),
            activation='sigmoid'
        )
        self.reg_conv2d = keras.layers.Conv2D(
            filters=4, kernel_size=3, strides=1, padding='same',
            kernel_initializer=_kernel_initializer,
            bias_initializer=tf.constant_initializer(0.1),
            activation='relu'
        )

        # Align Layer
        self.align_layer = AlignLayer(
            width, factor=config.HEAD_ALIGN_C, bias=config.HEAD_ALIGN_B
        )

        # Classification and Regression Reshape
        self.layer_reshape_cls = keras.layers.Reshape((-1, num_cls))
        self.layer_reshape_reg = keras.layers.Reshape((-1, 4))

    def call(self, inputs, training=None, mask=None):
        """
        :param inputs:   Shape=(None, None, width)
        :param training: None
        :param mask:     None
        :return:         Classification and Regression Subnetworks Output. (None, classes) and (None, 4)
        """

        # A Block for Regression subnet, (None, None, 256)
        a_block = inputs
        a_block = self.blocks_a(a_block)
        reg_out = self.reg_conv2d(a_block)

        # Align input, using centerness
        align_out = self.align_layer([reg_out, inputs])

        # B Block for Classification subnet(GN, 16 groups)
        align_out = self.blocks_b(align_out)
        cls_out = self.cls_conv2d(align_out)

        # Reshape Subnetwork's output
        cls_out_reshape = self.layer_reshape_cls(cls_out)
        reg_out_reshape = self.layer_reshape_reg(reg_out)
        return cls_out_reshape, reg_out_reshape

    def get_config(self):
        c = super(AlignHead, self).get_config()
        c.update(width=self.width)
        return c


def AlignSubnetworks(input_features, width=256, depth=4, num_cls=20):
    cls, reg = [], []

    subnetworks = AlignHead(
        width=width, depth=depth,
        num_cls=num_cls, ws=config.HEAD_WS
    )

    for feature in input_features:
        outputs = subnetworks(feature)
        cls.append(outputs[0])
        reg.append(outputs[1])

    cls_out = keras.layers.Concatenate(axis=1, name='align_cls_head')(cls)
    reg_out = keras.layers.Concatenate(axis=1, name='align_reg1_head')(reg)
    return cls_out, reg_out
