import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
import config
from tensorflow_addons.layers import GroupNormalization
from models.layers import WSConv2D


class PreLayer(keras.layers.Layer):
    def __init__(self, width=256, kernel_size=1, ws=0, *args, **kwargs):
        super(PreLayer, self).__init__(*args, **kwargs)

        self.conv2d = WSConv2D(
            filters=width, kernel_size=kernel_size, strides=1, padding='same',
            kernel_initializer=tf.initializers.RandomNormal(0.0, 0.01),
            bias_initializer='zeros'
        ) if ws else keras.layers.Conv2D(
            filters=width, kernel_size=kernel_size, strides=1, padding='same',
            kernel_initializer=tf.initializers.RandomNormal(0.0, 0.01),
            bias_initializer='zeros'
        )

        self.gn = GroupNormalization(groups=16, epsilon=1e-5)
        self.relu = keras.layers.Activation(tf.nn.relu)

    def call(self, inputs, **kwargs):
        x = self.conv2d(inputs)
        x = self.gn(x)
        x = self.relu(x)
        return x


@tf.function(jit_compile=True)
def _merge_method(l1, l2):
    return (l1 + l2) - (l1 * l2)


class MixCoreLayerV2(keras.layers.Layer):
    def __init__(self, kernel_size=3, ws=0, act=1, *args, **kwargs):
        super(MixCoreLayerV2, self).__init__(*args, **kwargs)

        _conv2d_config = {
            'filters': 256,
            'kernel_size': kernel_size,
            'strides': 1,
            'padding': 'same',
            'kernel_initializer': tf.initializers.RandomNormal(0.0, 0.01)
        }

        self.conv2d_reg = keras.Sequential()
        self.conv2d_cls = keras.Sequential()

        if ws:
            self.conv2d_reg.add(WSConv2D(**_conv2d_config))
            self.conv2d_cls.add(WSConv2D(**_conv2d_config))
        else:
            self.conv2d_reg.add(keras.layers.Conv2D(**_conv2d_config))
            self.conv2d_cls.add(keras.layers.Conv2D(**_conv2d_config))

        self.conv2d_reg.add(GroupNormalization(groups=16, epsilon=1e-5))
        self.conv2d_cls.add(GroupNormalization(groups=16, epsilon=1e-5))

        if act:
            self.conv2d_reg.add(keras.layers.Activation(tf.nn.relu))
            self.conv2d_cls.add(keras.layers.Activation(tf.nn.relu))

        self.conv2d_out = keras.layers.Conv2D(
            filters=256,
            kernel_size=1,
            strides=1,
            padding='same',
            kernel_initializer=tf.initializers.RandomNormal(0.0, 0.01)
        )

    def call(self, inputs, **kwargs):
        x = _merge_method(self.conv2d_reg(inputs[0]), self.conv2d_cls(inputs[1]))
        x = self.conv2d_out(x)
        return x


class MixHead(keras.Model):
    def __init__(self, width, depth, num_cls, ws=0, *args, **kwargs):
        super(MixHead, self).__init__(*args, **kwargs)

        _conv2d_setting = {
            'filters': width,
            'kernel_size': 3,
            'strides': 1,
            'padding': 'same',
            'kernel_initializer': tf.initializers.RandomNormal(0.0, 0.01)
        }

        self.reg_blocks = keras.Sequential()
        for _ in range(depth):
            self.reg_blocks.add(keras.layers.Conv2D(**_conv2d_setting))
            self.reg_blocks.add(keras.layers.Activation(tf.nn.relu))

        self.cls_blocks = keras.Sequential()
        for _ in range(depth):
            if ws:
                self.cls_blocks.add(WSConv2D(**_conv2d_setting))
            else:
                self.cls_blocks.add(keras.layers.Conv2D(**_conv2d_setting))
            self.cls_blocks.add(GroupNormalization(groups=16, epsilon=1e-5))
            self.cls_blocks.add(keras.layers.Activation(tf.nn.relu))

        # Mix Core
        self.p_layer_reg = PreLayer(width=width, kernel_size=3, ws=0)
        self.p_layer_cls = PreLayer(width=width, kernel_size=3, ws=0)

        self.cls_conv2d = keras.layers.Conv2D(
            filters=num_cls, kernel_size=3, strides=1, padding='same',
            kernel_initializer=tf.initializers.RandomNormal(0.0, 0.01),
            bias_initializer=tf.constant_initializer(-np.log((1 - 0.01) / 0.01)),
            activation='sigmoid')
        self.reg_conv2d = keras.layers.Conv2D(
            filters=4, kernel_size=3, strides=1, padding='same',
            kernel_initializer=tf.initializers.RandomNormal(0.0, 0.01),
            bias_initializer=tf.constant_initializer(0.1),
            activation='relu')

        self.layer_reshape_cls = keras.layers.Reshape((-1, num_cls))
        self.layer_reshape_reg = keras.layers.Reshape((-1, 4))

    def call(self, inputs, training=None, mask=None):
        # Regression block
        reg = self.reg_blocks(inputs)

        # Mix Core
        cls = _merge_method(self.p_layer_reg(reg), self.p_layer_cls(inputs))

        # Classification block
        cls = self.cls_blocks(cls)

        reg = self.reg_conv2d(reg)
        cls = self.cls_conv2d(cls)

        reg = self.layer_reshape_reg(reg)
        cls = self.layer_reshape_cls(cls)
        return cls, reg

    def get_config(self):
        pass


def MixSubnetworks(input_features, width=256, depth=4, num_cls=20):
    cls_, reg_ = [], []

    subnetworks = MixHead(width=width, depth=depth, num_cls=num_cls, ws=config.HEAD_WS)

    for feature in input_features:
        outputs = subnetworks(feature)
        cls_.append(outputs[0])
        reg_.append(outputs[1])

    cls_pred = keras.layers.Concatenate(axis=1, name='Classification')(cls_)
    reg_pred = keras.layers.Concatenate(axis=1, name='Regression')(reg_)
    return cls_pred, reg_pred
