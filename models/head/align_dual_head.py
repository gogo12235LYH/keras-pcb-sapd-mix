import tensorflow as tf
from tensorflow_addons.layers import GroupNormalization
import numpy as np
import tensorflow.keras as keras
import config


def standardize_weight(kernel, eps):
    mean = tf.math.reduce_mean(kernel, axis=(0, 1, 2), keepdims=True)
    # std = tf.math.reduce_std(kernel, axis=(0, 1, 2), keepdims=True)
    std = tf.sqrt(tf.math.reduce_variance(kernel, axis=(0, 1, 2), keepdims=True) + 1e-12)
    return (kernel - mean) / (std + eps)


class WSConv2D(tf.keras.layers.Conv2D):
    def __init__(self, *args, **kwargs):
        super(WSConv2D, self).__init__(*args, **kwargs)

    def call(self, inputs, eps=1e-5):
        self.kernel.assign(standardize_weight(self.kernel, eps))
        return super().call(inputs)


def _layer_normalization_V1(input_layer, gn=0, bn=0, bcn=0, groups=32):
    if gn:
        output_layer = GroupNormalization(groups=groups, epsilon=1e-3)(input_layer)

    elif bn:
        output_layer = keras.layers.BatchNormalization(axis=3, epsilon=1e-3)(input_layer)

    elif bcn:
        output_layer = keras.layers.BatchNormalization(axis=3, epsilon=1e-3)(input_layer)
        output_layer = GroupNormalization(groups=groups, epsilon=1e-3)(output_layer)

    else:
        output_layer = input_layer

    return output_layer


def _layer_centerness(reg_input, exp_dims=True):
    align_ = tf.minimum(reg_input[..., 0], reg_input[..., 2]) * tf.minimum(reg_input[..., 1], reg_input[..., 3]) / \
             tf.maximum(reg_input[..., 0], reg_input[..., 2]) / tf.maximum(reg_input[..., 1], reg_input[..., 3])

    if exp_dims:
        return tf.expand_dims(align_, axis=-1)

    else:
        return align_


def _align_block(reg_input, merge_input, width, factor):
    align_block_setting = {
        'filters': width,
        'kernel_size': 1,
        'strides': 1,
        'padding': 'same',
        'kernel_initializer': tf.initializers.RandomNormal(0.0, 0.01)
    }
    align_centerness = _layer_centerness(reg_input) * factor
    align_centerness = tf.repeat(align_centerness, width, axis=-1)
    align_out = keras.layers.Multiply()([align_centerness, merge_input])

    # align_out = keras.layers.Lambda(lambda x: x[0] * x[1])([align_centerness, merge_input])

    align_out = keras.layers.Conv2D(**align_block_setting)(align_out)

    return keras.layers.Add()([align_out, merge_input])


def _build_align_head(input_width=256, width=256, depth=4, num_cls=20, a_factor=0.25, b_factor=1.0, **kwargs):
    setting = {
        'kernel_size': 3,
        'strides': 1,
        'padding': 'same',
        'kernel_initializer': tf.initializers.RandomNormal(mean=0.0, stddev=0.01)
    }

    inputs = keras.layers.Input((None, None, input_width))

    a_blocks = inputs
    for i in range(depth):
        a_blocks = keras.layers.Conv2D(
            filters=width,
            activation='relu',
            bias_initializer='zeros',
            **setting
        )(a_blocks)

    reg_out_1 = keras.layers.Conv2D(
        filters=4,
        bias_initializer=tf.constant_initializer(0.1),
        activation='relu',
        name='reg_output_1',
        **setting
    )(a_blocks)
    reg_out_reshape_1 = keras.layers.Reshape((-1, 4), name='reg_reshape_1')(reg_out_1)

    # Align :
    align_out_reg_a = _align_block(reg_out_1, inputs, width, a_factor)

    b_blocks = align_out_reg_a
    for i in range(depth):
        b_blocks = keras.layers.Conv2D(
            filters=int(width * 1),
            bias_initializer='zeros',
            name=f'cls_a{i}',
            **setting
        )(b_blocks)
        b_blocks = _layer_normalization_V1(input_layer=b_blocks, **kwargs)
        b_blocks = keras.layers.Activation('relu')(b_blocks)

    # Align :
    align_out_reg_b = _align_block(reg_out_1, b_blocks, width, b_factor)

    cls_out = keras.layers.Conv2D(
        filters=num_cls,
        bias_initializer=tf.constant_initializer(-np.log((1 - 0.01) / 0.01)),
        activation='sigmoid',
        name='cls_output',
        **setting
    )(align_out_reg_b)
    cls_out_reshape = keras.layers.Reshape((-1, num_cls), name='cls_reshape')(cls_out)

    return keras.models.Model(inputs, outputs=[cls_out_reshape, reg_out_reshape_1],
                              name='align_head_model')


def AlignSubnetworks(input_features, width=256, depth=4, num_cls=20):
    cls, reg_1 = [], []

    align_head = _build_align_head(
        input_width=width, width=width, depth=depth, num_cls=num_cls,
        gn=1, groups=config.HEAD_GROUPS,
    )

    for feature in input_features:
        outputs = align_head(feature)
        cls.append(outputs[0])
        reg_1.append(outputs[1])

    cls_out = keras.layers.Concatenate(axis=1, name='align_cls_head')(cls)
    reg_out_1 = keras.layers.Concatenate(axis=1, name='align_reg1_head')(reg_1)

    # return cls_out, reg_out_2
    return cls_out, reg_out_1
