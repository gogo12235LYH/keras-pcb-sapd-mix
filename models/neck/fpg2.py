import tensorflow as tf
import tensorflow.keras as keras
from models import layers
from tensorflow_addons.layers import GroupNormalization
import tensorflow.keras.backend as k
from utils import util_graph


class UpsampleLike(keras.layers.Layer):
    """
        FPN's up-sample-like layers.

        Src : RetinaNet, https://github.com/fizyr/keras-retinanet

        inputs[0]: Src
        inputs[1]: Target
        outputs shape = Target
    """

    def call(self, inputs, **kwargs):
        src, target = inputs
        target_shape = k.shape(target)
        return util_graph.resize_images(src, (target_shape[1], target_shape[2]), method='nearest')

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],) + input_shape[1][1:3] + (input_shape[0][-1])


def __layer_normalization_V1(input_layer, gn=0, bn=0, bcn=0, groups=32, epsilon=1e-3):
    if gn:
        output_layer = GroupNormalization(groups=groups, epsilon=epsilon)(input_layer)

    elif bn:
        output_layer = keras.layers.BatchNormalization(epsilon=epsilon)(input_layer)

    elif bcn:
        output_layer = keras.layers.BatchNormalization(epsilon=epsilon)(input_layer)
        output_layer = GroupNormalization(groups=groups, epsilon=epsilon)(output_layer)

    else:
        output_layer = input_layer

    return output_layer


def _across_same_block(input_layer, act=1, num_channel=256, **kwargs):
    # R-C-B
    output_layer = input_layer
    # R
    if act:
        output_layer = keras.layers.Activation('relu')(output_layer)

    # C
    output_layer = keras.layers.Conv2D(
        filters=num_channel,
        kernel_size=1,
        strides=1,
        padding='same',
        # kernel_regularizer=layers.ws_reg
    )(output_layer)

    # B
    output_layer = __layer_normalization_V1(input_layer=output_layer, bn=1)
    return output_layer


def _across_up_block(input_layer, method='conv', num_channel=256, **kwargs):
    output_layer = input_layer
    output_layer = keras.layers.Activation('relu')(output_layer)

    if method == 'avg':
        output_layer = keras.layers.AvgPool2D(
            strides=2,
            padding='same',
        )(output_layer)

        return output_layer

    elif method == 'max':
        output_layer = keras.layers.MaxPool2D(
            strides=2,
            padding='same',
        )(output_layer)

        return output_layer

    elif method == 'conv':
        output_layer = keras.layers.Conv2D(
            filters=num_channel,
            kernel_size=3,
            strides=2,
            padding='same',
        )(output_layer)

        return output_layer

    else:
        raise ValueError('[FPG] Sum-Up Block Setting Wrong ! ')


def _across_down_block(input_layer, scale_factor=2, num_channel=256, **kwargs):
    h, w = tf.shape(input_layer)[1], tf.shape(input_layer)[2]
    h, w = h * scale_factor, w * scale_factor
    output_layer = tf.image.resize(input_layer, size=[h, w])

    output_layer = keras.layers.Activation('relu')(output_layer)

    output_layer = keras.layers.Conv2D(
        filters=num_channel,
        kernel_size=3,
        strides=1,
        padding='same',
        # kernel_regularizer=layers.ws_reg
    )(output_layer)

    output_layer = __layer_normalization_V1(input_layer=output_layer, bn=1)
    return output_layer


def _across_down_block_V2(input_layer_1, input_layer_2, num_channel=256, **kwargs):
    output_layer = UpsampleLike()([input_layer_1, input_layer_2])
    output_layer = keras.layers.Activation('relu')(output_layer)
    output_layer = keras.layers.Conv2D(
        filters=num_channel,
        kernel_size=3,
        strides=1,
        padding='same',
        # kernel_regularizer=layers.ws_reg
    )(output_layer)

    output_layer = __layer_normalization_V1(input_layer=output_layer, bn=1)
    return output_layer


def _across_skip_block(input_layer, num_channel=256, **kwargs):
    output_layer = input_layer
    output_layer = keras.layers.Activation('relu')(output_layer)

    output_layer = keras.layers.Conv2D(
        filters=num_channel,
        kernel_size=1,
        strides=1,
        padding='same',
        # kernel_regularizer=layers.ws_reg
    )(output_layer)

    output_layer = __layer_normalization_V1(input_layer=output_layer, bn=1)
    return output_layer


def _preprocess_fpg_V1(backbone_outputs, num_channel):
    """
    :param backbone_outputs:
    :param num_channel:
    :return:

        P7:  - -> P6 -> P7
        P6:  - -> C5 -> P6
        P5: C5 -> P5_l + P4_d -> P5
        P4: C4 -> P4_l + P3_d -> P4
        P3: C3 -> P3

    """
    C3, C4, C5 = backbone_outputs

    setting = {
        'filters': num_channel,
        'kernel_size': 1,
        'strides': 1,
        'padding': 'same'
    }

    down_setting = {
        'filters': num_channel,
        'kernel_size': 3,
        'strides': 2,
        'padding': 'same'
    }
    P3 = keras.layers.Conv2D(**setting, name='C3_reduced')(C3)  # P3

    P4_l = keras.layers.Conv2D(**setting, name='C4_reduced')(C4)
    P3_d = keras.layers.Conv2D(**down_setting, name='P3_down')(P3)
    P4 = keras.layers.Add(name='P4_Add')([P4_l, P3_d])  # P4

    P5_l = keras.layers.Conv2D(**setting, name='C5_reduced')(C5)
    P4_d = keras.layers.Conv2D(**down_setting, name='P4_down')(P4)
    P5 = keras.layers.Add(name='P5_Add')([P5_l, P4_d])  # P4

    P6 = keras.layers.Conv2D(**down_setting, name='P6')(P5)  # P6
    P6_relu = keras.layers.Activation('relu', name='P6_relu')(P6)

    P7 = keras.layers.Conv2D(**down_setting, name='P7')(P6_relu)  # P7

    return P3, P4, P5, P6, P7


def _preprocess_fpg_V2(backbone_outputs, num_channel):
    C3, C4, C5 = backbone_outputs

    setting = {
        'filters': num_channel,
        'strides': 1,
        'padding': 'same'
    }

    down_setting = {
        'filters': num_channel,
        'kernel_size': 3,
        'strides': 2,
        'padding': 'same'
    }

    P5 = keras.layers.Conv2D(kernel_size=1, **setting, name='C5_reduced')(C5)
    P5_upsampled = layers.UpsampleLike(name='P5_upsampled')([P5, C4])

    P4 = keras.layers.Conv2D(kernel_size=1, **setting, name='C4_reduced')(C4)
    P4 = keras.layers.Add(name='P4_merge')([P5_upsampled, P4])
    P4_upsampled = layers.UpsampleLike(name='P4_upsampled')([P4, C3])

    P3 = keras.layers.Conv2D(kernel_size=1, **setting, name='C3_reduced')(C3)
    P3 = keras.layers.Add(name='P3_merge')([P4_upsampled, P3])

    P6 = keras.layers.Conv2D(**down_setting, name='P6')(C5)  # P6
    P6_relu = keras.layers.Activation('relu', name='P6_relu')(P6)

    P7 = keras.layers.Conv2D(**down_setting, name='P7')(P6_relu)  # P7

    return P3, P4, P5, P6, P7


def _preprocess_fpg_V3(backbone_outputs, num_channel):
    """
    :param backbone_outputs:
    :param num_channel:
    :return:

        P7:  - -> P6 -> P7
        P6:  - -> C5 -> P6
        P5: C5 -> P5_l + P4_d -> P5
        P4: C4 -> P4_l + P3_d -> P4
        P3: C3 -> P3

    """
    C3, C4, C5 = backbone_outputs

    setting = {
        'filters': num_channel,
        'kernel_size': 1,
        'strides': 1,
        'padding': 'same'
    }

    down_setting = {
        'filters': num_channel,
        'kernel_size': 3,
        'strides': 2,
        'padding': 'same'
    }
    P3 = keras.layers.Conv2D(**setting, name='C3_reduced')(C3)  # P3

    P4_l = keras.layers.Conv2D(**setting, name='C4_reduced')(C4)
    P3_d = keras.layers.Conv2D(**down_setting, name='P3_down')(P3)
    P4 = keras.layers.Add(name='P4_Add')([P4_l, P3_d])  # P4

    P5_l = keras.layers.Conv2D(**setting, name='C5_reduced')(C5)
    P4_d = keras.layers.Conv2D(**down_setting, name='P4_down')(P4)
    P5 = keras.layers.Add(name='P5_Add')([P5_l, P4_d])  # P4

    P6 = keras.layers.Conv2D(**down_setting, name='P6')(P5)  # P6
    P6_relu = keras.layers.Activation('relu', name='P6_relu')(P6)

    P7 = keras.layers.Conv2D(**down_setting, name='P7')(P6_relu)  # P7

    return P3, P4, P5, P6, P7


def _fpg_level_1(last_features, num_channel=256, method='max'):
    P3_last, P4_last, P5_last, P6_last, P7_last = last_features

    # P6_AD = _across_down_block(input_layer=P6_last, scale_factor=2, num_channel=num_channel)
    P6_AD = _across_down_block_V2(input_layer_1=P6_last, input_layer_2=P5_last, num_channel=num_channel)
    P5_OUT = keras.layers.Add()([P5_last, P6_AD])  # P5
    P5_UP = _across_up_block(input_layer=P5_OUT, method=method, num_channel=num_channel)

    P6_L = _across_same_block(input_layer=P6_last, act=1, num_channel=num_channel)
    # P7_AD = _across_down_block(input_layer=P7_last, scale_factor=2, num_channel=num_channel)
    P7_AD = _across_down_block_V2(input_layer_1=P7_last, input_layer_2=P6_last, num_channel=num_channel)
    P6_SK = _across_skip_block(input_layer=P6_last, num_channel=num_channel)
    P6_OUT = keras.layers.Add()([P5_UP, P6_L, P7_AD, P6_SK])  # P6
    P6_UP = _across_up_block(input_layer=P6_OUT, method=method, num_channel=num_channel)

    P7_L = _across_same_block(input_layer=P7_last, act=1, num_channel=num_channel)
    P7_SK = _across_skip_block(input_layer=P7_last, num_channel=num_channel)
    P7_OUT = keras.layers.Add()([P6_UP, P7_L, P7_SK])  # P7

    return P3_last, P4_last, P5_OUT, P6_OUT, P7_OUT


def _fpg_level_2(org_feature, last_features, num_channel=256, method='max'):
    P3_org, P4_org, P5_org, P6_org, P7_org = org_feature
    P3_last, P4_last, P5_last, P6_last, P7_last = last_features

    # P5_AD = _across_down_block(input_layer=P5_last, scale_factor=2, num_channel=num_channel)
    P5_AD = _across_down_block_V2(input_layer_1=P5_last, input_layer_2=P4_last, num_channel=num_channel)
    P4_SK = _across_skip_block(input_layer=P4_org, num_channel=num_channel)
    P4_OUT = keras.layers.Add()([P4_last, P5_AD, P4_SK])  # P4
    P4_UP = _across_up_block(input_layer=P4_OUT, method=method, num_channel=num_channel)

    P5_L = _across_same_block(input_layer=P5_last, act=1, num_channel=num_channel)
    # P6_AD = _across_down_block(input_layer=P6_last, scale_factor=2, num_channel=num_channel)
    P6_AD = _across_down_block_V2(input_layer_1=P6_last, input_layer_2=P5_last, num_channel=num_channel)
    P5_SK = _across_skip_block(input_layer=P5_org, num_channel=num_channel)
    P5_OUT = keras.layers.Add()([P4_UP, P5_L, P6_AD, P5_SK])  # P5
    P5_UP = _across_up_block(input_layer=P5_OUT, method=method, num_channel=num_channel)

    P6_L = _across_same_block(input_layer=P6_last, act=1, num_channel=num_channel)
    # P7_AD = _across_down_block(input_layer=P7_last, scale_factor=2, num_channel=num_channel)
    P7_AD = _across_down_block_V2(input_layer_1=P7_last, input_layer_2=P6_last, num_channel=num_channel)
    P6_SK = _across_skip_block(input_layer=P6_org, num_channel=num_channel)
    P6_OUT = keras.layers.Add()([P5_UP, P6_L, P7_AD, P6_SK])  # P6
    P6_UP = _across_up_block(input_layer=P6_OUT, method=method, num_channel=num_channel)

    P7_L = _across_same_block(input_layer=P7_last, act=1, num_channel=num_channel)
    P7_SK = _across_skip_block(input_layer=P7_org, num_channel=num_channel)
    P7_OUT = keras.layers.Add()([P6_UP, P7_L, P7_SK])  # P7

    return P3_last, P4_OUT, P5_OUT, P6_OUT, P7_OUT


def _fpg_level_3(org_feature, last_features, num_channel=256, method='max'):
    P3_org, P4_org, P5_org, P6_org, P7_org = org_feature
    P3_last, P4_last, P5_last, P6_last, P7_last = last_features

    # P4_AD = _across_down_block(input_layer=P4_last, scale_factor=2, num_channel=num_channel)
    P4_AD = _across_down_block_V2(input_layer_1=P4_last, input_layer_2=P3_last, num_channel=num_channel)
    P3_SK = _across_skip_block(input_layer=P3_org, num_channel=num_channel)
    P3_OUT = keras.layers.Add()([P3_last, P4_AD, P3_SK])  # P3
    P3_UP = _across_up_block(input_layer=P3_OUT, method=method, num_channel=num_channel)

    P4_L = _across_same_block(input_layer=P4_last, act=1, num_channel=num_channel)
    # P5_AD = _across_down_block(input_layer=P5_last, scale_factor=2, num_channel=num_channel)
    P5_AD = _across_down_block_V2(input_layer_1=P5_last, input_layer_2=P4_last, num_channel=num_channel)
    P4_SK = _across_skip_block(input_layer=P4_org, num_channel=num_channel)
    P4_OUT = keras.layers.Add()([P3_UP, P4_L, P5_AD, P4_SK])  # P4
    P4_UP = _across_up_block(input_layer=P4_OUT, method=method, num_channel=num_channel)

    P5_L = _across_same_block(input_layer=P5_last, act=1, num_channel=num_channel)
    # P6_AD = _across_down_block(input_layer=P6_last, scale_factor=2, num_channel=num_channel)
    P6_AD = _across_down_block_V2(input_layer_1=P6_last, input_layer_2=P5_last, num_channel=num_channel)
    P5_SK = _across_skip_block(input_layer=P5_org, num_channel=num_channel)
    P5_OUT = keras.layers.Add()([P4_UP, P5_L, P6_AD, P5_SK])  # P5
    P5_UP = _across_up_block(input_layer=P5_OUT, method=method, num_channel=num_channel)

    P6_L = _across_same_block(input_layer=P6_last, act=1, num_channel=num_channel)
    # P7_AD = _across_down_block(input_layer=P7_last, scale_factor=2, num_channel=num_channel)
    P7_AD = _across_down_block_V2(input_layer_1=P7_last, input_layer_2=P6_last, num_channel=num_channel)
    P6_SK = _across_skip_block(input_layer=P6_org, num_channel=num_channel)
    P6_OUT = keras.layers.Add()([P5_UP, P6_L, P7_AD, P6_SK])  # P6
    P6_UP = _across_up_block(input_layer=P6_OUT, method=method, num_channel=num_channel)

    P7_L = _across_same_block(input_layer=P7_last, act=1, num_channel=num_channel)
    P7_SK = _across_skip_block(input_layer=P7_org, num_channel=num_channel)
    P7_OUT = keras.layers.Add()([P6_UP, P7_L, P7_SK])  # P7

    return P3_OUT, P4_OUT, P5_OUT, P6_OUT, P7_OUT


def cp_add_method(input_layers):
    def cp(input_xs):
        output_x_sum = 0.
        output_x_mul = 0.

        for input_x in input_xs:
            output_x_sum += input_x
            output_x_mul *= input_x

        return output_x_sum - output_x_mul

    return keras.layers.Lambda(lambda x: cp(x))(input_layers)


def _fpg_level_n(org_feature, last_features, num_channel=256, method='max'):
    P3_org, P4_org, P5_org, P6_org, P7_org = org_feature
    P3_last, P4_last, P5_last, P6_last, P7_last = last_features

    P3_L = _across_same_block(input_layer=P3_last, act=1, num_channel=num_channel)
    # P4_AD = _across_down_block(input_layer=P4_last, scale_factor=2, num_channel=num_channel)
    P4_AD = _across_down_block_V2(input_layer_1=P4_last, input_layer_2=P3_last, num_channel=num_channel)
    P3_SK = _across_skip_block(input_layer=P3_org, num_channel=num_channel)
    P3_OUT = keras.layers.Add()([P3_L, P4_AD, P3_SK])  # P3
    # P3_OUT = cp_add_method([P3_L, P4_AD, P3_SK])
    P3_UP = _across_up_block(input_layer=P3_OUT, method=method, num_channel=num_channel)

    P4_L = _across_same_block(input_layer=P4_last, act=1, num_channel=num_channel)
    # P5_AD = _across_down_block(input_layer=P5_last, scale_factor=2, num_channel=num_channel)
    P5_AD = _across_down_block_V2(input_layer_1=P5_last, input_layer_2=P4_last, num_channel=num_channel)
    P4_SK = _across_skip_block(input_layer=P4_org, num_channel=num_channel)
    P4_OUT = keras.layers.Add()([P3_UP, P4_L, P5_AD, P4_SK])  # P4
    # P4_OUT = cp_add_method([P3_UP, P4_L, P5_AD, P4_SK])
    P4_UP = _across_up_block(input_layer=P4_OUT, method=method, num_channel=num_channel)

    P5_L = _across_same_block(input_layer=P5_last, act=1, num_channel=num_channel)
    # P6_AD = _across_down_block(input_layer=P6_last, scale_factor=2, num_channel=num_channel)
    P6_AD = _across_down_block_V2(input_layer_1=P6_last, input_layer_2=P5_last, num_channel=num_channel)
    P5_SK = _across_skip_block(input_layer=P5_org, num_channel=num_channel)
    P5_OUT = keras.layers.Add()([P4_UP, P5_L, P6_AD, P5_SK])  # P5
    # P5_OUT = cp_add_method([P4_UP, P5_L, P6_AD, P5_SK])
    P5_UP = _across_up_block(input_layer=P5_OUT, method=method, num_channel=num_channel)

    P6_L = _across_same_block(input_layer=P6_last, act=1, num_channel=num_channel)
    # P7_AD = _across_down_block(input_layer=P7_last, scale_factor=2, num_channel=num_channel)
    P7_AD = _across_down_block_V2(input_layer_1=P7_last, input_layer_2=P6_last, num_channel=num_channel)
    P6_SK = _across_skip_block(input_layer=P6_org, num_channel=num_channel)
    P6_OUT = keras.layers.Add()([P5_UP, P6_L, P7_AD, P6_SK])  # P6
    # P6_OUT = cp_add_method([P5_UP, P6_L, P7_AD, P6_SK])
    P6_UP = _across_up_block(input_layer=P6_OUT, method=method, num_channel=num_channel)

    P7_L = _across_same_block(input_layer=P7_last, act=1, num_channel=num_channel)
    P7_SK = _across_skip_block(input_layer=P7_org, num_channel=num_channel)
    P7_OUT = keras.layers.Add()([P6_UP, P7_L, P7_SK])  # P7
    # P7_OUT = cp_add_method([P6_UP, P7_L, P7_SK])

    return P3_OUT, P4_OUT, P5_OUT, P6_OUT, P7_OUT


def _build_fpg(backbone_outputs, num_channel=256, method='max', level=5):
    # Original FPN.
    org_features = _preprocess_fpg_V2(
        backbone_outputs=backbone_outputs,
        num_channel=num_channel
    )

    last_features = _fpg_level_1(
        last_features=org_features,
        num_channel=num_channel,
        method=method
    )
    last_features = _fpg_level_2(
        last_features=last_features,
        org_feature=org_features,
        num_channel=num_channel,
        method=method
    )
    last_features = _fpg_level_3(
        last_features=last_features,
        org_feature=org_features,
        num_channel=num_channel,
        method=method
    )

    if level > 3:
        for _ in range(level - 3):
            last_features = _fpg_level_n(
                last_features=last_features, org_feature=org_features,
                num_channel=num_channel,
                method=method
            )

    output_features = []

    for feature in last_features:
        output_features.append(
            keras.layers.Conv2D(
                filters=num_channel,
                kernel_size=3,
                strides=1,
                padding='same',
            )(feature)
        )

    return output_features


if __name__ == '__main__':
    backbone_outputs_ = []

    img = 512

    for i in range(3):
        fm = int(512 / int(2 ** (i + 2)))

        backbone_outputs_.append(
            keras.layers.Input(shape=(fm, fm, 256))
        )

    org_features_ = _build_fpg(backbone_outputs=backbone_outputs_, num_channel=128, level=9)

    models = keras.models.Model(backbone_outputs_, org_features_, )
    models.summary()
