import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from tensorflow_addons.layers import GroupNormalization


class StdHead(keras.Model):
    def __init__(self, width, depth, num_cls, gn=1):
        super(StdHead, self).__init__()

        self.width = width
        self.depth = depth
        self.num_cls = num_cls
        self.gn = gn

        _conv2d_setting = {
            'filters': self.width,
            'kernel_size': 3,
            'strides': 1,
            'padding': 'same',
            'kernel_initializer': tf.initializers.RandomNormal(0.0, 0.01),
        }

        self.cls_blocks = []
        for _ in range(self.depth):
            self.cls_blocks.append(keras.layers.Conv2D(**_conv2d_setting, groups=16))

            if gn:
                self.cls_blocks.append(GroupNormalization(groups=16))

            self.cls_blocks.append(keras.layers.Activation(tf.nn.relu))

        self.reg_blocks = []
        for _ in range(self.depth):
            self.reg_blocks.append(keras.layers.Conv2D(**_conv2d_setting))
            self.reg_blocks.append(keras.layers.Activation(tf.nn.relu))

        self.cls_conv2d = keras.layers.Conv2D(
            filters=num_cls, kernel_size=3, strides=1, padding='same',
            kernel_initializer=tf.initializers.RandomNormal(0.0, 0.01),
            bias_initializer=tf.constant_initializer(-np.log((1 - 0.01) / 0.01)),
            activation='sigmoid'
        )
        self.reg_conv2d = keras.layers.Conv2D(
            filters=4, kernel_size=3, strides=1, padding='same',
            kernel_initializer=tf.initializers.RandomNormal(0.0, 0.01),
            bias_initializer=tf.constant_initializer(0.1),
            activation='relu'
        )

        self.cls_reshape = keras.layers.Reshape((-1, self.num_cls))
        self.reg_reshape = keras.layers.Reshape((-1, 4))

    def call(self, inputs, training=None, mask=None):
        cls = inputs
        reg = inputs

        for cls_layer in self.cls_blocks:
            cls = cls_layer(cls)
        for reg_layer in self.reg_blocks:
            reg = reg_layer(reg)

        cls = self.cls_conv2d(cls)
        reg = self.reg_conv2d(reg)

        cls = self.cls_reshape(cls)
        reg = self.reg_reshape(reg)
        return cls, reg

    def get_config(self):
        c_fig = super(StdHead, self).get_config()
        c_fig.update({
            "width": self.width,
            "depth": self.depth,
            "num_cls": self.num_cls,
            "gn": self.gn
        })
        return c_fig


def StdSubnetworks(input_features, width=256, depth=4, num_cls=20):
    cls, reg = [], []

    subnetworks = StdHead(width=width, depth=depth, num_cls=num_cls)

    for feature in input_features:
        outputs = subnetworks(feature)
        cls.append(outputs[0])
        reg.append(outputs[1])

    cls_out = keras.layers.Concatenate(axis=1, name='cls_head')(cls)
    reg_out = keras.layers.Concatenate(axis=1, name='reg_head')(reg)
    return cls_out, reg_out
