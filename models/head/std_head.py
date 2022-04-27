import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from tensorflow_addons.layers import GroupNormalization


class StandardHead(keras.layers.Layer):
    def __init__(self, width, depth, num_cls, gn=1, name='Std_head', **kwargs):
        super(StandardHead, self).__init__(name=name, **kwargs)

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

        self.cls_blocks = keras.Sequential()
        self.reg_blocks = keras.Sequential()
        for _ in range(self.depth):
            self.cls_blocks.add(keras.layers.Conv2D(**_conv2d_setting))
            self.reg_blocks.add(keras.layers.Conv2D(**_conv2d_setting))
            if gn:
                self.cls_blocks.add(GroupNormalization(groups=32))
            self.cls_blocks.add(keras.layers.ReLU())
            self.reg_blocks.add(keras.layers.ReLU())

        self.cls_conv2d = keras.layers.Conv2D(
            filters=num_cls, kernel_size=3, strides=1, padding='same',
            kernel_initializer=tf.initializers.RandomNormal(0.0, 0.01),
            bias_initializer=tf.constant_initializer(-np.log((1 - 0.01) / 0.01)),
        )
        self.reg_conv2d = keras.layers.Conv2D(
            filters=4, kernel_size=3, strides=1, padding='same',
            kernel_initializer=tf.initializers.RandomNormal(0.0, 0.01),
            bias_initializer=tf.constant_initializer(0.1),
        )

        self.cls_reshape = keras.layers.Reshape((-1, self.num_cls))
        self.reg_reshape = keras.layers.Reshape((-1, 4))

    def call(self, inputs, training=None, mask=None):
        cls = inputs
        reg = inputs

        cls = self.cls_blocks(cls)
        reg = self.reg_blocks(reg)

        cls = self.cls_conv2d(cls)
        reg = self.reg_conv2d(reg)

        cls = keras.layers.Activation('sigmoid', dtype='float32')(cls)
        reg = keras.layers.Activation('relu', dtype='float32')(reg)

        cls = self.cls_reshape(cls)
        reg = self.reg_reshape(reg)
        return cls, reg

    def get_config(self):
        c_fig = super(StandardHead, self).get_config()
        c_fig.update(
            {
                "width": self.width,
                "depth": self.depth,
                "num_cls": self.num_cls,
                "gn": self.gn
            }
        )
        return c_fig


def StdSubnetworks(input_features, width=256, depth=4, num_cls=20):
    cls, reg = [], []

    subnetworks = StandardHead(width=width, depth=depth, num_cls=num_cls)

    for feature in input_features:
        outputs = subnetworks(feature)
        cls.append(outputs[0])
        reg.append(outputs[1])

    cls_out = keras.layers.Concatenate(axis=1, name='cls_head')(cls)
    reg_out = keras.layers.Concatenate(axis=1, name='reg_head')(reg)
    return cls_out, reg_out


class Subnetworks(keras.Model):
    def __init__(self, width=256, depth=4, num_cls=20, name='StdHead', **kwargs):
        super(Subnetworks, self).__init__(name=name, **kwargs)

        self.width = width
        self.depth = depth
        self.num_cls = num_cls

        self.head = StandardHead(width=width, depth=depth, num_cls=num_cls, gn=1)

    def call(self, inputs, training=None, mask=None):
        cls, reg = [], []

        for input_features in inputs:
            out = self.head(input_features)
            cls.append(out[0])
            reg.append(out[1])

        cls_out = keras.layers.Concatenate(axis=1, name='cls_head')(cls)
        reg_out = keras.layers.Concatenate(axis=1, name='reg_head')(reg)

        return cls_out, reg_out

    def get_config(self):
        cfg = super(Subnetworks, self).get_config()
        cfg.update(
            {
                'width': self.width,
                'depth': self.depth,
                'num_cls': self.num_cls,
            }
        )
