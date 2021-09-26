import tensorflow as tf
import numpy as np
from tensorflow import keras


class StdHead(keras.Model):
    def __init__(self, width, depth, num_cls):
        super(StdHead, self).__init__()

        _conv2d_setting = {
            'filters': width,
            'kernel_size': 3,
            'strides': 1,
            'padding': 'same',
            'kernel_initializer': tf.initializers.RandomNormal(0.0, 0.01),
        }

        self.cls_blocks = keras.Sequential()
        for _ in range(depth):
            self.cls_blocks.add(keras.layers.Conv2D(**_conv2d_setting))
            self.cls_blocks.add(keras.layers.Activation(tf.nn.relu))

        self.reg_blocks = keras.Sequential()
        for _ in range(depth):
            self.reg_blocks.add(keras.layers.Conv2D(**_conv2d_setting))
            self.reg_blocks.add(keras.layers.Activation(tf.nn.relu))

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

        self.cls_reshape = keras.layers.Reshape((-1, num_cls))
        self.reg_reshape = keras.layers.Reshape((-1, 4))

    def call(self, inputs, training=None, mask=None):
        cls = self.cls_blocks(inputs)
        reg = self.reg_blocks(inputs)

        cls = self.cls_conv2d(cls)
        reg = self.reg_conv2d(reg)

        cls = self.cls_reshape(cls)
        reg = self.reg_reshape(reg)
        return cls, reg

    def get_config(self):
        return super(StdHead, self).get_config()


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
