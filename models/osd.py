from functools import reduce
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from models import layers
from models.layers import ws_reg
from tensorflow_addons.layers import GroupNormalization
import math
import keras_resnet
import keras_resnet.models
import numpy as np

MOMENTUM = 0.997
EPSILON = 1e-4

w_bi_fpn = [64, 88, 112, 160, 224, 288, 384, 256]
d_bi_fpn = [3, 4, 5, 6, 7, 7, 8, 4]


class PriorProbability(keras.initializers.Initializer):
    """
    Focal loss : Retinanet paper上的初始化設定
    """

    def __init__(self, probability=0.01):
        self.probability = probability

    def get_config(self):
        return {
            'probability': self.probability
        }

    def __call__(self, shape, dtype=None):
        return tf.ones(shape=shape, dtype=dtype) * -math.log((1 - self.probability) / self.probability)


class FixedValueBiasInitializer(keras.initializers.Initializer):
    def __init__(self, value):
        self.value = value

    def get_config(self):
        return {
            'value': self.value
        }

    def __call__(self, shape, dtype=None):
        # set bias
        result = tf.ones(shape, dtype=np.float32) * self.value
        return result


def _create_pyramid_features(C3, C4, C5, feature_size=256, n=0):
    """
    :param C3:
    :param C4:
    :param C5:
    :param feature_size:
    :return:
    """
    # C3 : (None, 80, 80, 128)
    # C4 : (None, 40, 40, 256)
    # C5 : (None, 20, 20, 512)

    setting = {
        'filters': feature_size,
        'strides': 1,
        'padding': 'same'
    }

    down_setting = {
        'filters': feature_size,
        'kernel_size': 3,
        'strides': 2,
        'padding': 'same'
    }

    # P5 : (None, 20, 20, 256)
    P5 = keras.layers.Conv2D(kernel_size=1, **setting, name='C5_reduced')(C5)
    # P5_upsampled : (None, 40, 40, 256)
    P5_upsampled = layers.UpsampleLike(name='P5_upsampled')([P5, C4])
    # P5_upsampled_2 = dilated_block(input_layer=P5_upsampled, width=feature_size, name='P5')
    # P5_upsampled_2 = layers.UpsampleLike(name='P5_upsampled_2')([P5_upsampled_2, C4])
    # P5 : (None, 20, 20, 256)
    P5 = keras.layers.Conv2D(kernel_size=3, **setting, name='P5', kernel_regularizer=ws_reg if n else None)(P5)

    # P4 : (None, 40, 40, 256)
    P4 = keras.layers.Conv2D(kernel_size=1, **setting, name='C4_reduced')(C4)
    P4 = keras.layers.Add(name='P4_merge')([P5_upsampled, P4])
    # P4_upsampled : (None, 80, 80, 256)
    P4_upsampled = layers.UpsampleLike(name='P4_upsampled')([P4, C3])
    # P4_upsampled_2 = dilated_block(input_layer=P4_upsampled, width=feature_size, name='P4')
    # P4_upsampled_2 = layers.UpsampleLike(name='P4_upsampled_2')([P4_upsampled_2, C3])
    # P4 : (None, 40, 40, 256)
    P4 = keras.layers.Conv2D(kernel_size=3, **setting, name='P4', kernel_regularizer=ws_reg if n else None)(P4)

    # P3 : (None, 80, 80, 256)
    P3 = keras.layers.Conv2D(kernel_size=1, **setting, name='C3_reduced')(C3)
    P3 = keras.layers.Add(name='P3_merge')([P4_upsampled, P3])
    P3 = keras.layers.Conv2D(kernel_size=3, **setting, name='P3', kernel_regularizer=ws_reg if n else None)(P3)

    # P6 : (None, 10, 10, 256)
    P6 = keras.layers.Conv2D(**down_setting, name='P6', kernel_regularizer=ws_reg if n else None)(C5)
    if n:
        P6 = GroupNormalization(name='P6_gn')(P6)
    P6_relu = keras.layers.Activation('relu', name='P6_relu')(P6)

    # P7 : (None, 5, 5, 256)
    P7 = keras.layers.Conv2D(**down_setting, name='P7', kernel_regularizer=ws_reg if n else None)(P6_relu)

    if n:
        P7 = GroupNormalization(name='P7_gn')(P7)
        P5 = GroupNormalization(name='P5_gn')(P5)
        P4 = GroupNormalization(name='P4_gn')(P4)
        P3 = GroupNormalization(name='P3_gn')(P3)

    return [P3, P4, P5, P6, P7]


def dilated_block(input_layer, width, name, d=3, split=0):
    output_layer = input_layer
    setting = {
        'filters': int(width / d) if split else width,
        'strides': 1,
        'padding': 'same',
        'kernel_size': 3
    }

    for i in range(d):
        output_layer = keras.layers.Conv2D(**setting, dilation_rate=i + 1, name=name + f'_d{i + 1}')(output_layer)

    return output_layer


def _group_block(features, channel=256):
    outputs = []
    for feature in features:
        outputs.append(
            keras.layers.Conv2D(
                filters=channel,
                kernel_size=3,
                strides=1,
                padding='same'
            )(feature)
        )
    return outputs


def _create_dense_pyramid_features_V1(C3, C4, C5, feature_size=256):
    setting = {
        'filters': feature_size,
        'strides': 1,
        'padding': 'same'
    }

    down_setting = {
        'filters': feature_size,
        'kernel_size': 3,
        'strides': 2,
        'padding': 'same'
    }
    P3 = keras.layers.Conv2D(kernel_size=1, **setting, name='C3_reduced')(C3)
    P4 = keras.layers.Conv2D(kernel_size=1, **setting, name='C4_reduced')(C4)
    P5 = keras.layers.Conv2D(kernel_size=1, **setting, name='C5_reduced')(C5)

    P6 = keras.layers.Conv2D(**down_setting, name='P6')(C5)
    P6_relu = keras.layers.Activation('relu', name='P6_relu')(P6)

    P7 = keras.layers.Conv2D(**down_setting, name='P7')(P6_relu)

    P5 = keras.layers.Conv2D(kernel_size=3, **setting, name='P5')(P5)

    P4_f5 = layers.UpsampleLike(name='P4_f5')([P5, C4])
    P4 = keras.layers.Concatenate(axis=-1, name='P4_concat')([P4, P4_f5])
    P4 = keras.layers.Conv2D(kernel_size=3, **setting, name='P4')(P4)

    P3_f5 = layers.UpsampleLike(name='P3_f5')([P5, C3])
    P3_f4 = layers.UpsampleLike(name='P3_f4')([P4, C3])
    P3 = keras.layers.Concatenate(axis=-1, name='P3_concat')([P3, P3_f5, P3_f4])
    P3 = keras.layers.Conv2D(kernel_size=3, **setting, name='P3')(P3)
    return [P3, P4, P5, P6, P7]


def _create_dense_pyramid_features_V2(C3, C4, C5, feature_size=256, merge_method='add'):
    setting = {
        'filters': feature_size,
        'strides': 1,
        'padding': 'same'
    }

    down_setting = {
        'filters': feature_size,
        'kernel_size': 3,
        'strides': 2,
        'padding': 'same'
    }

    P3 = keras.layers.Conv2D(kernel_size=1, **setting, name='C3_reduced')(C3)
    P4 = keras.layers.Conv2D(kernel_size=1, **setting, name='C4_reduced')(C4)
    P5 = keras.layers.Conv2D(kernel_size=1, **setting, name='C5_reduced')(C5)

    P6 = keras.layers.Conv2D(**down_setting, name='P6')(C5)
    P6_relu = keras.layers.Activation('relu', name='P6_relu')(P6)

    P7 = keras.layers.Conv2D(**down_setting, name='P7')(P6_relu)

    P5_f7 = layers.UpsampleLike(name='P5_f7')([P7, C5])
    P5_f6 = layers.UpsampleLike(name='P5_f6')([P6, C5])

    if merge_method == 'concat':
        P5 = keras.layers.Concatenate(axis=-1, name='P5_concat')([P5, P5_f7, P5_f6])
    elif merge_method == 'add':
        P5 = keras.layers.Add(name='P5_add')([P5, P5_f7, P5_f6])
    else:
        P5 = keras.layers.Add(name='P5_add')([P5, P5_f7, P5_f6])

    P5 = keras.layers.Conv2D(kernel_size=3, **setting, name='P5')(P5)

    P4_f7 = layers.UpsampleLike(name='P4_f7')([P7, C4])
    P4_f6 = layers.UpsampleLike(name='P4_f6')([P6, C4])
    P4_f5 = layers.UpsampleLike(name='P4_f5')([P5, C4])

    if merge_method == 'concat':
        P4 = keras.layers.Concatenate(axis=-1, name='P4_concat')([P4, P4_f7, P4_f6, P4_f5])
    elif merge_method == 'add':
        P4 = keras.layers.Add(name='P4_add')([P4, P4_f7, P4_f6, P4_f5])
    else:
        P4 = keras.layers.Add(name='P4_add')([P4, P4_f7, P4_f6, P4_f5])

    P4 = keras.layers.Conv2D(kernel_size=3, **setting, name='P4')(P4)

    P3_f7 = layers.UpsampleLike(name='P3_f7')([P7, C3])
    P3_f6 = layers.UpsampleLike(name='P3_f6')([P6, C3])
    P3_f5 = layers.UpsampleLike(name='P3_f5')([P5, C3])
    P3_f4 = layers.UpsampleLike(name='P3_f4')([P4, C3])

    if merge_method == 'concat':
        P3 = keras.layers.Concatenate(axis=-1, name='P3_concat')([P3, P3_f7, P3_f6, P3_f5, P3_f4])
    elif merge_method == 'add':
        P3 = keras.layers.Add(name='P3_add')([P3, P3_f7, P3_f6, P3_f5, P3_f4])
    else:
        P3 = keras.layers.Add(name='P3_add')([P3, P3_f7, P3_f6, P3_f5, P3_f4])

    P3 = keras.layers.Conv2D(kernel_size=3, **setting, name='P3')(P3)
    return [P3, P4, P5, P6, P7]


def _down_sampling(input_layer, number=1):
    strides = int(2 ** number)
    output_layer = keras.layers.MaxPooling2D(
        (strides, strides),
        padding='same',
        strides=strides
    )(input_layer)
    return output_layer


def _gp_block(inputs_layer1, inputs_layer2):
    # inputs_layer1: small-Level
    # inputs_layer2: large-Level
    # outputs_layer: large-Level

    outputs_layer1 = tf.math.reduce_max(inputs_layer1, axis=[1, 2], keepdims=True)
    outputs_layer1 = keras.layers.Activation('sigmoid')(outputs_layer1)

    outputs_layer = keras.layers.Lambda(lambda x: x[0] * x[1])([outputs_layer1, inputs_layer2])

    inputs_layer1_upsample = layers.UpsampleLike()([inputs_layer1, inputs_layer2])

    outputs_layer = keras.layers.Add()([outputs_layer, inputs_layer1_upsample])

    return outputs_layer


def _rcb_block(inputs_layer, num_channel=256):
    outputs_layer = keras.layers.Activation('relu')(inputs_layer)
    outputs_layer = keras.layers.Conv2D(
        filters=num_channel,
        kernel_size=3,
        strides=1,
        padding='same'
    )(outputs_layer)
    outputs_layer = keras.layers.BatchNormalization(epsilon=1e-5)(outputs_layer)
    return outputs_layer


def _sum_block(inputs_layer1, inputs_layer2):
    inputs_layer1_resize = layers.UpsampleLike()([inputs_layer1, inputs_layer2])
    output = keras.layers.Add()([inputs_layer1_resize, inputs_layer2])
    return output


def SeparableConvBlock(num_channels, kernel_size, strides, name, freeze_b=False):
    f1 = keras.layers.SeparableConv2D(num_channels, kernel_size=kernel_size, strides=strides, padding='same',
                                      use_bias=True, name=f'{name}/conv')
    f2 = keras.layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, name=f'{name}/bn')
    return reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), (f1, f2))


def intermediate_node(inputs,
                      num_channels, kernel_size, strides, blocks,
                      node_name, combine_name):
    output_td = keras.layers.Add(name=f'fpn_cells/cell_{blocks}/f_noe{node_name[0]}/add')(inputs)
    output_td = keras.layers.Activation(lambda x: tf.nn.swish(x))(output_td)
    return SeparableConvBlock(num_channels=num_channels,
                              kernel_size=kernel_size,
                              strides=strides,
                              name=f'fpn_cells/cell_{blocks}/f_node{node_name[1]}/op_after_{combine_name}'
                              )(output_td)


def resample_conv2d(input, num_channels, kernel_size=1, padding='same', blocks=None, node_num=None, name=None):
    output = keras.layers.Conv2D(num_channels, kernel_size=kernel_size, padding='same',
                                 name=f'fpn_cells/cell_{blocks}/f_node{node_num}/resample{name}/conv2d')(input)
    output = keras.layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                             name=f'fpn_cells/cell_{blocks}/f_node{node_num}/resample{name}/bn')(output)
    return output


def _create_Bi_Feature_Pyramid(inputs, num_channels=256, blocks=0):
    if blocks == 0:
        _, _, C3, C4, C5 = inputs
        P3_in, P4_in, P5_in = C3, C4, C5

        P6_in = keras.layers.Conv2D(num_channels, kernel_size=1, padding='same', name='resample_p6/conv2d')(C5)
        P6_in = keras.layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, name='resample_p6/bn')(P6_in)
        P6_in = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name='resample_p6/maxpool')(P6_in)

        P7_in = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name='resample_p7/maxpool')(P6_in)

        # P7_U = keras.layers.UpSampling2D()(P7_in)
        P7_U = layers.UpsampleLike(name='P7_upsampled')([P7_in, P6_in])
        P6_td = intermediate_node(inputs=[P6_in, P7_U], num_channels=num_channels, kernel_size=3, strides=1,
                                  blocks=blocks, node_name=['0', '1'], combine_name='combine5')

        P5_in_1 = resample_conv2d(input=P5_in, num_channels=num_channels, kernel_size=1, padding='same',
                                  blocks=blocks, node_num='1', name='_0_2_6'
                                  )

        # P6_U = keras.layers.UpSampling2D()(P6_td)
        P6_U = layers.UpsampleLike(name='P6_upsampled')([P6_td, P5_in_1])
        P5_td = intermediate_node(inputs=[P5_in_1, P6_U], num_channels=num_channels, kernel_size=3, strides=1,
                                  blocks=blocks, node_name=['1', '1'], combine_name='combine6'
                                  )

        P4_in_1 = resample_conv2d(input=P4_in, num_channels=num_channels, kernel_size=1, padding='same',
                                  blocks=blocks, node_num='2', name='_0_1_7'
                                  )

        # P5_U = keras.layers.UpSampling2D()(P5_td)
        P5_U = layers.UpsampleLike(name='P5_upsampled')([P5_td, P4_in_1])
        P4_td = intermediate_node(inputs=[P4_in_1, P5_U], num_channels=num_channels, kernel_size=3, strides=1,
                                  blocks=blocks, node_name=['2', '3'], combine_name='combine7'
                                  )

        P3_in = resample_conv2d(input=P3_in, num_channels=num_channels, kernel_size=1, padding='same',
                                blocks=blocks, node_num='3', name='_0_0_8'
                                )

        # P4_U = keras.layers.UpSampling2D()(P4_td)
        P4_U = layers.UpsampleLike(name='P4_upsampled')([P4_td, P3_in])
        P3_out = intermediate_node(inputs=[P3_in, P4_U], num_channels=num_channels, kernel_size=3, strides=1,
                                   blocks=blocks, node_name=['3', '3'], combine_name='combine8'
                                   )

        P4_in_2 = resample_conv2d(input=P4_in, num_channels=num_channels, kernel_size=1, padding='same',
                                  blocks=blocks, node_num='4', name='_0_1_9'
                                  )

        P3_D = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P3_out)
        P4_out = intermediate_node(inputs=[P4_in_2, P4_td, P3_D], num_channels=num_channels, kernel_size=3, strides=1,
                                   blocks=blocks, node_name=['4', '4'], combine_name='combine9'
                                   )

        P5_in_2 = resample_conv2d(input=P5_in, num_channels=num_channels, kernel_size=1, padding='same',
                                  blocks=blocks, node_num='5', name='_0_2_10'
                                  )

        P4_D = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P4_out)
        P5_out = intermediate_node(inputs=[P5_in_2, P5_td, P4_D], num_channels=num_channels, kernel_size=3, strides=1,
                                   blocks=blocks, node_name=['5', '5'], combine_name='combine10'
                                   )

        P5_D = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P5_out)
        P6_out = intermediate_node(inputs=[P6_in, P6_td, P5_D], num_channels=num_channels, kernel_size=3, strides=1,
                                   blocks=blocks, node_name=['6', '6'], combine_name='combine11'
                                   )

        P6_D = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P6_out)
        P7_out = intermediate_node(inputs=[P7_in, P6_D], num_channels=num_channels, kernel_size=3, strides=1,
                                   blocks=blocks, node_name=['7', '7'], combine_name='combine12'
                                   )

    else:
        P3_in, P4_in, P5_in, P6_in, P7_in = inputs

        # P7_U = keras.layers.UpSampling2D()(P7_in)
        P7_U = layers.UpsampleLike(name=f'cells_{blocks}/P7_upsampled')([P7_in, P6_in])
        P6_td = intermediate_node(inputs=[P7_U, P6_in], num_channels=num_channels, kernel_size=3, strides=1,
                                  blocks=blocks, node_name=['0', '0'], combine_name='combine5'
                                  )

        # P6_U = keras.layers.UpSampling2D()(P6_td)
        P6_U = layers.UpsampleLike(name=f'cells_{blocks}/P6_upsampled')([P6_td, P5_in])
        P5_td = intermediate_node(inputs=[P6_U, P5_in], num_channels=num_channels, kernel_size=3, strides=1,
                                  blocks=blocks, node_name=['1', '1'], combine_name='combine6'
                                  )

        # P5_U = keras.layers.UpSampling2D()(P5_td)
        P5_U = layers.UpsampleLike(name=f'cells_{blocks}/P5_upsampled')([P5_td, P4_in])
        P4_td = intermediate_node(inputs=[P5_U, P4_in], num_channels=num_channels, kernel_size=3, strides=1,
                                  blocks=blocks, node_name=['2', '2'], combine_name='combine7'
                                  )

        # P4_U = keras.layers.UpSampling2D()(P4_td)
        P4_U = layers.UpsampleLike(name=f'cells_{blocks}/P4_upsampled')([P4_td, P3_in])
        P3_out = intermediate_node(inputs=[P4_U, P3_in], num_channels=num_channels, kernel_size=3, strides=1,
                                   blocks=blocks, node_name=['3', '3'], combine_name='combine8'
                                   )

        P3_D = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P3_out)
        P4_out = intermediate_node(inputs=[P4_in, P4_td, P3_D], num_channels=num_channels, kernel_size=3, strides=1,
                                   blocks=blocks, node_name=['4', '4'], combine_name='combine9'
                                   )

        P4_D = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P4_out)
        P5_out = intermediate_node(inputs=[P5_in, P5_td, P4_D], num_channels=num_channels, kernel_size=3, strides=1,
                                   blocks=blocks, node_name=['5', '5'], combine_name='combine10'
                                   )

        P5_D = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P5_out)
        P6_out = intermediate_node(inputs=[P6_in, P6_td, P5_D], num_channels=num_channels, kernel_size=3, strides=1,
                                   blocks=blocks, node_name=['6', '6'], combine_name='combine11'
                                   )

        P6_D = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P6_out)
        P7_out = intermediate_node(inputs=[P7_in, P6_D], num_channels=num_channels, kernel_size=3, strides=1,
                                   blocks=blocks, node_name=['7', '7'], combine_name='combine12'
                                   )

    return P3_out, P4_out, P5_out, P6_out, P7_out


def _build_pyramid_header(model_name, model, features):
    """
    :param model_name: sub-model's name
    :param model: sub-model
    :param features: [P3, P4, P5, P6, P7]
    :return:
    """
    # If input image's shape is (640, 640, 3)
    # P3 : (80, 80, 256)
    # P4 : (40, 40, 256)
    # P5 : (20, 20, 256)
    # P6 : (10, 10, 256)
    # P7 : (5, 5, 256)

    # regression sub-model output : (B, 8525, 4)
    # classification sub-model output : (B, 8525, num_cls)
    # center-ness sub-model output : (B, 8525, 1)
    return keras.layers.Concatenate(axis=1, name=model_name)([model(f) for f in features])


def _build_final_pyramid(models, features):
    """
    :param models: sub-models' tuple, which is (model_name) and (model).
    :param features: [P3, P4, P5, P6, P7] Feature Pyramid Network.
    :return: A list of [(B, 8525, 4), (B, 8525, num_cls), (B, 8525, 1)]
    """
    return [_build_pyramid_header(model_name, model, features) for model_name, model in models]


def _build_location(features):
    """
    :return:
    """
    locations = layers.Locations(strides=[8, 16, 32, 64, 128], name='locations')(features)
    return locations


def d_shared_mode(input_feature_size=256,
                  model_feature_size=256,
                  depth=4,
                  name='shared_model'):
    """
    :param input_feature_size:
    :param model_feature_size:
    :param name:
    :param depth:
    :return:
    """

    setting = {
        'kernel_size': 3,
        'strides': 1,
        'padding': 'same',
        'kernel_initializer': keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
        'bias_initializer': 'zeros'
    }

    inputs = keras.Input(shape=(None, None, input_feature_size))
    # inputs = keras.Input(shape=(None, None, None))
    outputs = inputs

    for i in range(depth):
        outputs = keras.layers.Conv2D(filters=model_feature_size,
                                      activation='relu',
                                      name='shared_model_{}'.format(i),
                                      # groups=28,
                                      **setting
                                      )(outputs)
        # outputs = keras.layers.Activation(lambda x: tf.nn.swish(x))(outputs)

    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)


def d_classification_model(num_cls,
                           shared_model,
                           input_feature_size=256,
                           name='classification_submodel'):
    """

    Retinanet中的classification sub-model。

    :param num_cls:
    :param shared_model:
    :param input_feature_size:
    :param name:
    :return:
    """
    # If inputs shape : (80, 80, 256)
    inputs = keras.layers.Input(shape=(None, None, input_feature_size))
    # inputs = keras.layers.Input(shape=(None, None, None))

    # (None, 80, 80, 256)
    outputs = shared_model(inputs)

    # (None, 80, 80, num_cls)
    outputs = keras.layers.Conv2D(filters=num_cls,
                                  kernel_size=3,
                                  strides=1,
                                  padding='same',
                                  kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
                                  bias_initializer=PriorProbability(probability=0.01),
                                  name='cls_1'
                                  )(outputs)

    # (None, 80*80, 6) -> (None, 6400, 6)
    outputs = keras.layers.Reshape((-1, num_cls), name='cls_Reshape')(outputs)
    outputs = keras.layers.Activation('sigmoid', name='cls_Output')(outputs)
    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)


def d_centerness_model(shared_model,
                       input_feature_size=256,
                       name='centerness_submodel'):
    """
    :param name:
        Centerness Sub-model
    :param shared_model:
        共享模型(四層深度為256之模型)
    :param input_feature_size:
        輸入層的深度大小
    :return:
    """
    # If inputs shape : (80, 80, 256)
    inputs = keras.layers.Input(shape=(None, None, input_feature_size))
    # inputs = keras.layers.Input(shape=(None, None, None))

    # (None, 80, 80, 256)
    outputs = shared_model(inputs)

    # (None, 80, 80, 1)
    outputs = keras.layers.Conv2D(filters=1,
                                  kernel_size=3,
                                  strides=1,
                                  padding='same',
                                  kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
                                  bias_initializer='zeros',
                                  name='centerness_1'
                                  )(outputs)

    # (None, 80*80, 1) -> (None, 6400, 1)
    outputs = keras.layers.Reshape(target_shape=(-1, 1), name='centerness_Reshape')(outputs)
    outputs = keras.layers.Activation('sigmoid', name='centerness_Output')(outputs)

    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)


def d_regression_model(num_values=4,
                       input_feature_size=256,
                       model_feature_size=256,
                       depth=4,
                       name='regression_submodel'):
    """
    :arg:
        num_values : 輸出預測值數量
        input_feature_size : 輸入此model的filter數量
        model_feature_size : model中隱藏層的filter數量

    :return:
        回傳keras.models.Model的模型資訊
    """

    setting = {
        'kernel_size': 3,
        'strides': 1,
        'padding': 'same',
        'kernel_initializer': keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
        'bias_initializer': 'zeros'
    }

    # If inputs shape : (80, 80, 256)
    inputs = keras.layers.Input(shape=(None, None, input_feature_size))
    outputs = inputs

    # 回歸模型建立
    # (None, 80, 80, 256)
    for i in range(depth):
        outputs = keras.layers.Conv2D(filters=model_feature_size,
                                      activation='relu',
                                      name='reg_{}'.format(i),
                                      **setting
                                      )(outputs)
        # outputs = keras.layers.Activation(lambda x: tf.nn.swish(x))(outputs)

    # 頂層前處理
    # (None, 80, 80, num_values)
    outputs = keras.layers.Conv2D(filters=num_values,
                                  activation='relu',
                                  name='reg_output',
                                  **setting
                                  )(outputs)

    # (None, 80*80, num_values) -> (None, 6400, num_values)
    outputs = keras.layers.Reshape(target_shape=(-1, num_values),
                                   name='reg_reshape_output')(outputs)

    # mapping value from 0 to info, from FCOS.
    outputs = keras.layers.Lambda(lambda x: K.exp(x))(outputs)
    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)


def d_sub_model(num_cls, width=256, submodel_width=256, depth=4):
    shared_model = d_shared_mode(input_feature_size=width,
                                 model_feature_size=submodel_width,
                                 depth=depth
                                 )

    return [
        ('regression', d_regression_model(num_values=4,
                                          input_feature_size=width,
                                          model_feature_size=submodel_width,
                                          depth=depth
                                          )),
        ('classification', d_classification_model(num_cls=num_cls,
                                                  shared_model=shared_model,
                                                  input_feature_size=width,
                                                  )),
        ('centerness', d_centerness_model(shared_model=shared_model,
                                          input_feature_size=width,
                                          ))
    ]


def create_base_model(inputs, backbone_output, num_cls, name='FCOS'):
    C3, C4, C5 = backbone_output
    sub_model = d_sub_model(num_cls)

    # [P3, P4, P5, P6, P7]
    features = _create_pyramid_features(C3, C4, C5)
    head = _build_final_pyramid(sub_model, features)

    return keras.models.Model(inputs=inputs, outputs=head, name=name)


def create_base_model_w_BiFPN(inputs, backbone_output, num_cls, blocks=3, name='FCOS_w_BiFPN'):
    fpn_feature = [None, None, backbone_output[0], backbone_output[1], backbone_output[2]]

    w_bi = w_bi_fpn[blocks]
    d_bi = d_bi_fpn[blocks]
    # w_bi, d_bi = 256, 4

    sub_model = d_sub_model(num_cls=num_cls, width=w_bi, submodel_width=w_bi, depth=d_bi)
    # sub_model = d_sub_model(num_cls=num_cls, width=w_bi, depth=4)

    for i in range(d_bi):
        fpn_feature = _create_Bi_Feature_Pyramid(fpn_feature, num_channels=w_bi, blocks=i)

    head = _build_final_pyramid(sub_model, fpn_feature)

    return keras.models.Model(inputs=inputs, outputs=head, name=name)


def inference_model(model, nms=True, class_specific_filter=True, w_BiFPN=False, blocks=2):
    """

    """
    # if blocks = 2, blocks_output_name = 5 - 1 = 4
    blocks_output_name = d_bi_fpn[blocks] - 1

    bi_fpn_layers_name = [
        f'fpn_cells/cell_{blocks_output_name}/f_node3/op_after_combine8/bn',
        f'fpn_cells/cell_{blocks_output_name}/f_node4/op_after_combine9/bn',
        f'fpn_cells/cell_{blocks_output_name}/f_node5/op_after_combine10/bn',
        f'fpn_cells/cell_{blocks_output_name}/f_node6/op_after_combine11/bn',
        f'fpn_cells/cell_{blocks_output_name}/f_node7/op_after_combine12/bn',
    ]

    if w_BiFPN is False:
        features = [model.get_layer(p_name).output for p_name in ['P3', 'P4', 'P5', 'P6', 'P7']]

    else:
        features = [model.get_layer(p_name).output for p_name in bi_fpn_layers_name]

    regression = model.outputs[0]
    classification = model.outputs[1]
    centerness = model.outputs[2]

    locations = _build_location(features)

    boxes = layers.RegressionBoxes(name='boxes')([locations, regression])
    boxes = layers.ClipBoxes(name='clipped_boxes')([model.inputs[0], boxes])

    detections = layers.FilterDetections(
        nms=nms,
        class_specific_filter=class_specific_filter,
        name='filtered_detections'
    )([boxes, classification, centerness])

    return keras.models.Model(inputs=model.inputs, outputs=detections, name='inference_model')


def test_backbone(inputs):
    setting = {
        'filters': 128,
        'kernel_size': 3,
        'strides': 2,
        'padding': 'same'
    }
    # (320, 320, 64)
    C1 = keras.layers.Conv2D(**setting, name='C1')(inputs)
    C1_relu = keras.layers.Activation('relu')(C1)

    # (160, 160, 64)
    C2 = keras.layers.Conv2D(**setting, name='C2')(C1_relu)
    C2_relu = keras.layers.Activation('relu')(C2)

    # (80, 80, 128)
    C3 = keras.layers.Conv2D(**setting, name='C3')(C2_relu)
    C3_relu = keras.layers.Activation('relu')(C3)

    # (40, 40, 256)
    C4 = keras.layers.Conv2D(**setting, name='C4')(C3_relu)
    C4_relu = keras.layers.Activation('relu')(C4)

    # (20, 20, 512)
    C5 = keras.layers.Conv2D(**setting, name='C5')(C4_relu)
    C5_relu = keras.layers.Activation('relu')(C5)
    return [C3, C4, C5]


def test_build(num_cls=6, mode=True):
    inputs = keras.layers.Input(shape=(None, None, 3))

    if mode is False:
        backbone_output = test_backbone(inputs)

    else:
        resnet = keras_resnet.models.ResNet50(inputs=inputs,
                                              include_top=False,
                                              freeze_bn=True
                                              )
        backbone_output = resnet.outputs[1:]

    test_model = create_base_model(inputs, backbone_output, num_cls, name='TEST')

    return test_model

# if __name__ == '__main__':
#     inputs_ = keras.layers.Input(shape=(None, None, 3))
#     resnet_ = keras_resnet.models.ResNet101(inputs=inputs_,
#                                             include_top=False,
#                                             freeze_bn=True
#                                             )
#
#     model1 = create_base_model_w_BiFPN(inputs_, resnet_.outputs[1:], 20)
#     model1.summary()
#
#     x = model1.predict(np.ones((1, 600, 600, 3)))

# model2 = create_base_model(inputs_, resnet_.outputs[1:], 20)
# model2.summary()
