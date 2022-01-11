import tensorflow as tf
import numpy as np
import tensorflow.keras as keras


class CSP_Conv2d(keras.layers.Layer):
    def __init__(self, width=256, *args, **kwargs):
        super(CSP_Conv2d, self).__init__(width, *args, **kwargs)

        _1x1_conv2d_setting = {
            'kernel_size': 1,
            'strides': 1,
            'padding': 'same',
            'kernel_initializer': tf.initializers.RandomNormal(0.0, 0.01),
            'activation': tf.nn.relu
        }

        _3x3_conv2d_setting = {
            'filters': int(width / 2),
            'kernel_size': 3,
            'strides': 1,
            'padding': 'same',
            'kernel_initializer': tf.initializers.RandomNormal(0.0, 0.01),
            'activation': tf.nn.swish
        }

        self.p1_conv2d_f = keras.layers.Conv2D(filters=int(width / 2), **_1x1_conv2d_setting)
        self.p1_conv2d_l = keras.layers.Conv2D(filters=width, **_1x1_conv2d_setting)
        self.p2_conv2d = keras.layers.Conv2D(**_3x3_conv2d_setting)

    def call(self, inputs, *args, **kwargs):
        p1, p2 = tf.split(inputs, num_or_size_splits=2, axis=-1)

        p1 = self.p1_conv2d_f(p1)
        p2 = self.p2_conv2d(p2)

        out = keras.layers.Concatenate()[p1, p2]
        # out = self.p1_conv2d_l(out)
        return out


class StdCSPHead(keras.Model):
    def __init__(self, width, depth, num_cls):
        self.width = width
        self.depth = depth
        self.num_cls = num_cls
        super(StdCSPHead, self).__init__()

        self.cls_blocks = keras.Sequential()
        for _ in range(self.depth):
            self.cls_blocks(CSP_Conv2d(width=width))

        self.reg_blocks = keras.Sequential()
        for _ in range(self.depth):
            self.reg_blocks(CSP_Conv2d(width=width))

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
        cls = self.cls_blocks(inputs)
        reg = self.reg_blocks(inputs)

        cls = self.cls_conv2d(cls)
        reg = self.reg_conv2d(reg)

        cls = self.cls_reshape(cls)
        reg = self.reg_reshape(reg)
        return cls, reg

    def get_config(self):
        c_fig = super(StdCSPHead, self).get_config()
        c_fig.update({
            "width": self.width,
            "depth": self.depth,
            "num_cls": self.num_cls
        })
        return c_fig


def StdCSPSubnetworks(input_features, width=256, depth=4, num_cls=20):
    cls, reg = [], []

    subnetworks = StdCSPHead(width=width, depth=depth, num_cls=num_cls)

    for feature in input_features:
        outputs = subnetworks(feature)
        cls.append(outputs[0])
        reg.append(outputs[1])

    cls_out = keras.layers.Concatenate(axis=1, name='cls_head')(cls)
    reg_out = keras.layers.Concatenate(axis=1, name='reg_head')(reg)
    return cls_out, reg_out
