import tensorflow as tf
import tensorflow.keras as keras
from models import layers


def _across_same_block(input_layer, act=1, num_channel=256):
    # R-C-B
    output_layer = input_layer

    # Activation: R
    if act:
        output_layer = keras.layers.Activation('relu')(output_layer)

    # Convolution: C
    output_layer = keras.layers.Conv2D(
        filters=num_channel,
        kernel_size=1,
        strides=1,
        padding='same',
    )(output_layer)

    # BatchNormalization: B
    output_layer = keras.layers.BatchNormalization()(output_layer)
    return output_layer


def _across_up_block(input_layer, skip=0, num_channel=256):
    if skip:
        return keras.layers.MaxPool2D(
            # pool_size=3,
            strides=2,
            padding='same',
        )(input_layer)

    else:
        return keras.layers.Conv2D(
            filters=num_channel,
            kernel_size=3,
            strides=2,
            padding='same',
        )(input_layer)


def _across_down_block(input_layer, scale_factor=2, num_channel=256):
    h, w = tf.shape(input_layer)[1], tf.shape(input_layer)[2]
    h, w = h * scale_factor, w * scale_factor
    output_layer = tf.image.resize(input_layer, size=[h, w])

    return keras.layers.Conv2D(
        filters=num_channel,
        kernel_size=3,
        strides=1,
        padding='same',
    )(output_layer)


def _across_skip_block(input_layer, num_channel=256):
    return keras.layers.Conv2D(
        filters=num_channel,
        kernel_size=1,
        strides=1,
        padding='same',
    )(input_layer)


def _pre_fpg_process_V1(backbone_outputs, num_channel):
    down_setting = {
        'filters': num_channel,
        'kernel_size': 3,
        'strides': 2,
        'padding': 'same'
    }

    C2, C3, C4, C5 = backbone_outputs

    P2_L = _across_same_block(input_layer=C2, act=0, num_channel=num_channel)  # P2
    P3_L = _across_same_block(input_layer=C3, act=0, num_channel=num_channel)
    P4_L = _across_same_block(input_layer=C4, act=0, num_channel=num_channel)
    P5_L = _across_same_block(input_layer=C5, act=0, num_channel=num_channel)

    P3 = keras.layers.Conv2D(**down_setting)(P2_L)
    P3 = keras.layers.Add()([P3_L, P3])  # P3
    P3_act = keras.layers.Activation('relu')(P3)

    P4 = keras.layers.Conv2D(**down_setting)(P3_act)
    P4 = keras.layers.Add()([P4_L, P4])  # P4
    P4_act = keras.layers.Activation('relu')(P4)

    P5 = keras.layers.Conv2D(**down_setting)(P4_act)
    P5 = keras.layers.Add()([P5_L, P5])  # P5
    P5_act = keras.layers.Activation('relu')(P5)

    P6 = keras.layers.Conv2D(**down_setting)(P5_act)  # P6

    return P2_L, P3, P4, P5, P6


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
    P5 = keras.layers.Conv2D(kernel_size=3, **setting, name='P5')(P5)  # P5

    P4 = keras.layers.Conv2D(kernel_size=1, **setting, name='C4_reduced')(C4)
    P4 = keras.layers.Add(name='P4_merge')([P5_upsampled, P4])
    P4_upsampled = layers.UpsampleLike(name='P4_upsampled')([P4, C3])
    P4 = keras.layers.Conv2D(kernel_size=3, **setting, name='P4')(P4)  # P4

    P3 = keras.layers.Conv2D(kernel_size=1, **setting, name='C3_reduced')(C3)
    P3 = keras.layers.Add(name='P3_merge')([P4_upsampled, P3])
    P3 = keras.layers.Conv2D(kernel_size=3, **setting, name='P3')(P3)  # P3

    P6 = keras.layers.Conv2D(**down_setting, name='P6')(C5)  # P6
    P6_relu = keras.layers.Activation('relu', name='P6_relu')(P6)

    P7 = keras.layers.Conv2D(**down_setting, name='P7')(P6_relu)  # P7

    return P3, P4, P5, P6, P7


def _fpg_level_1(last_features, num_channel=256, skip=1):
    P3_last, P4_last, P5_last, P6_last, P7_last = last_features

    P7_L = _across_same_block(input_layer=P7_last, act=1, num_channel=num_channel)
    P6_L = _across_same_block(input_layer=P6_last, act=1, num_channel=num_channel)

    P7_AD = _across_down_block(input_layer=P7_last, scale_factor=2, num_channel=num_channel)
    P6_AD = _across_down_block(input_layer=P6_last, scale_factor=2, num_channel=num_channel)

    P6_SK = _across_skip_block(input_layer=P6_last, num_channel=num_channel)
    P7_SK = _across_skip_block(input_layer=P7_last, num_channel=num_channel)

    P5_OUT = keras.layers.Add()([P5_last, P6_AD])  # P5
    P5_UP = _across_up_block(input_layer=P5_OUT, skip=skip, num_channel=num_channel)

    P6_OUT = keras.layers.Add()([P5_UP, P6_L, P7_AD, P6_SK])  # P6
    P6_UP = _across_up_block(input_layer=P6_OUT, skip=skip, num_channel=num_channel)

    P7_OUT = keras.layers.Add()([P6_UP, P7_L, P7_SK])  # P7

    return P3_last, P4_last, P5_OUT, P6_OUT, P7_OUT


def _fpg_level_2(org_feature, last_features, num_channel=256, skip=1):
    P3_org, P4_org, P5_org, P6_org, P7_org = org_feature
    P3_last, P4_last, P5_last, P6_last, P7_last = last_features

    P7_L = _across_same_block(input_layer=P7_last, act=1, num_channel=num_channel)
    P6_L = _across_same_block(input_layer=P6_last, act=1, num_channel=num_channel)
    P5_L = _across_same_block(input_layer=P5_last, act=1, num_channel=num_channel)

    P7_AD = _across_down_block(input_layer=P7_last, scale_factor=2, num_channel=num_channel)
    P6_AD = _across_down_block(input_layer=P6_last, scale_factor=2, num_channel=num_channel)
    P5_AD = _across_down_block(input_layer=P5_last, scale_factor=2, num_channel=num_channel)

    P7_SK = _across_skip_block(input_layer=P7_org, num_channel=num_channel)
    P6_SK = _across_skip_block(input_layer=P6_org, num_channel=num_channel)
    P5_SK = _across_skip_block(input_layer=P5_org, num_channel=num_channel)
    P4_SK = _across_skip_block(input_layer=P4_org, num_channel=num_channel)

    P4_OUT = keras.layers.Add()([P4_last, P5_AD, P4_SK])  # P4
    P4_UP = _across_up_block(input_layer=P4_OUT, skip=skip, num_channel=num_channel)

    P5_OUT = keras.layers.Add()([P4_UP, P5_L, P6_AD, P5_SK])  # P5
    P5_UP = _across_up_block(input_layer=P5_OUT, skip=skip, num_channel=num_channel)

    P6_OUT = keras.layers.Add()([P5_UP, P6_L, P7_AD, P6_SK])  # P6
    P6_UP = _across_up_block(input_layer=P6_OUT, skip=skip, num_channel=num_channel)

    P7_OUT = keras.layers.Add()([P6_UP, P7_L, P7_SK])  # P7

    return P3_last, P4_OUT, P5_OUT, P6_OUT, P7_OUT


def _fpg_level_3(org_feature, last_features, num_channel=256, skip=1):
    P3_org, P4_org, P5_org, P6_org, P7_org = org_feature
    P3_last, P4_last, P5_last, P6_last, P7_last = last_features

    P7_L = _across_same_block(input_layer=P7_last, act=1, num_channel=num_channel)
    P6_L = _across_same_block(input_layer=P6_last, act=1, num_channel=num_channel)
    P5_L = _across_same_block(input_layer=P5_last, act=1, num_channel=num_channel)
    P4_L = _across_same_block(input_layer=P4_last, act=1, num_channel=num_channel)

    P7_AD = _across_down_block(input_layer=P7_last, scale_factor=2, num_channel=num_channel)
    P6_AD = _across_down_block(input_layer=P6_last, scale_factor=2, num_channel=num_channel)
    P5_AD = _across_down_block(input_layer=P5_last, scale_factor=2, num_channel=num_channel)
    P4_AD = _across_down_block(input_layer=P4_last, scale_factor=2, num_channel=num_channel)

    P7_SK = _across_skip_block(input_layer=P7_org, num_channel=num_channel)
    P6_SK = _across_skip_block(input_layer=P6_org, num_channel=num_channel)
    P5_SK = _across_skip_block(input_layer=P5_org, num_channel=num_channel)
    P4_SK = _across_skip_block(input_layer=P4_org, num_channel=num_channel)
    P3_SK = _across_skip_block(input_layer=P3_org, num_channel=num_channel)

    P3_OUT = keras.layers.Add()([P3_last, P4_AD, P3_SK])  # P3
    P3_UP = _across_up_block(input_layer=P3_OUT, skip=skip, num_channel=num_channel)

    P4_OUT = keras.layers.Add()([P3_UP, P4_L, P5_AD, P4_SK])  # P4
    P4_UP = _across_up_block(input_layer=P4_OUT, skip=skip, num_channel=num_channel)

    P5_OUT = keras.layers.Add()([P4_UP, P5_L, P6_AD, P5_SK])  # P5
    P5_UP = _across_up_block(input_layer=P5_OUT, skip=skip, num_channel=num_channel)

    P6_OUT = keras.layers.Add()([P5_UP, P6_L, P7_AD, P6_SK])  # P6
    P6_UP = _across_up_block(input_layer=P6_OUT, skip=skip, num_channel=num_channel)

    P7_OUT = keras.layers.Add()([P6_UP, P7_L, P7_SK])  # P7

    return P3_OUT, P4_OUT, P5_OUT, P6_OUT, P7_OUT


def _fpg_level_n(org_feature, last_features, num_channel=256, skip=1):
    P3_org, P4_org, P5_org, P6_org, P7_org = org_feature
    P3_last, P4_last, P5_last, P6_last, P7_last = last_features

    P7_L = _across_same_block(input_layer=P7_last, act=1, num_channel=num_channel)
    P6_L = _across_same_block(input_layer=P6_last, act=1, num_channel=num_channel)
    P5_L = _across_same_block(input_layer=P5_last, act=1, num_channel=num_channel)
    P4_L = _across_same_block(input_layer=P4_last, act=1, num_channel=num_channel)
    P3_L = _across_same_block(input_layer=P3_last, act=1, num_channel=num_channel)

    P7_AD = _across_down_block(input_layer=P7_last, scale_factor=2, num_channel=num_channel)
    P6_AD = _across_down_block(input_layer=P6_last, scale_factor=2, num_channel=num_channel)
    P5_AD = _across_down_block(input_layer=P5_last, scale_factor=2, num_channel=num_channel)
    P4_AD = _across_down_block(input_layer=P4_last, scale_factor=2, num_channel=num_channel)

    P7_SK = _across_skip_block(input_layer=P7_org, num_channel=num_channel)
    P6_SK = _across_skip_block(input_layer=P6_org, num_channel=num_channel)
    P5_SK = _across_skip_block(input_layer=P5_org, num_channel=num_channel)
    P4_SK = _across_skip_block(input_layer=P4_org, num_channel=num_channel)
    P3_SK = _across_skip_block(input_layer=P3_org, num_channel=num_channel)

    P3_OUT = keras.layers.Add()([P3_L, P4_AD, P3_SK])  # P3
    P3_UP = _across_up_block(input_layer=P3_OUT, skip=skip, num_channel=num_channel)

    P4_OUT = keras.layers.Add()([P3_UP, P4_L, P5_AD, P4_SK])  # P4
    P4_UP = _across_up_block(input_layer=P4_OUT, skip=skip, num_channel=num_channel)

    P5_OUT = keras.layers.Add()([P4_UP, P5_L, P6_AD, P5_SK])  # P5
    P5_UP = _across_up_block(input_layer=P5_OUT, skip=skip, num_channel=num_channel)

    P6_OUT = keras.layers.Add()([P5_UP, P6_L, P7_AD, P6_SK])  # P6
    P6_UP = _across_up_block(input_layer=P6_OUT, skip=skip, num_channel=num_channel)

    P7_OUT = keras.layers.Add()([P6_UP, P7_L, P7_SK])  # P7

    return P3_OUT, P4_OUT, P5_OUT, P6_OUT, P7_OUT


def _build_fpg(backbone_outputs, num_channel=256, skip=0, level=5):
    org_features = _preprocess_fpg_V2(backbone_outputs=backbone_outputs, num_channel=num_channel)

    last_features = _fpg_level_1(last_features=org_features, num_channel=num_channel, skip=skip)
    last_features = _fpg_level_2(last_features=last_features, org_feature=org_features, num_channel=num_channel,
                                 skip=skip)
    last_features = _fpg_level_3(last_features=last_features, org_feature=org_features, num_channel=num_channel,
                                 skip=skip)

    if level > 3:
        for _ in range(level-3):
            last_features = _fpg_level_n(last_features=last_features, org_feature=org_features, num_channel=num_channel
                                         , skip=skip)

    return last_features


if __name__ == '__main__':
    backbone_outputs_ = []

    img = 512

    for i in range(3):
        fm = int(512 / int(2 ** (i + 2)))

        backbone_outputs_.append(
            keras.layers.Input(shape=(fm, fm, 256))
        )

    org_features_ = _build_fpg(backbone_outputs=backbone_outputs_, num_channel=256)

    models = keras.models.Model(backbone_outputs_, org_features_, )
    models.summary()
