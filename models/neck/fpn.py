import tensorflow as tf
import tensorflow.keras as keras
from tensorflow_addons.layers import GroupNormalization
from models import layers

_k_init = tf.initializers.RandomNormal(0.0, 0.01)

_lateral_dict = {
    'filters': 256,
    'kernel_size': 1,
    'strides': 1,
    'padding': 'same',
}

_same_dict = {
    'filters': 256,
    'kernel_size': 3,
    'strides': 1,
    'padding': 'same',
}

_down_dict = {
    'filters': 256,
    'kernel_size': 3,
    'strides': 2,
    'padding': 'same',
}


class FPN(keras.Model):
    """
        From : https://keras.io/examples/vision/retinanet/
    """

    def __init__(self, interpolation='nearest', **kwargs):
        super(FPN, self).__init__(name='FPN', **kwargs)

        self.interpolation = interpolation

        # lateral branch
        self.l_conv2d_c3 = keras.layers.Conv2D(**_lateral_dict)
        self.l_conv2d_c4 = keras.layers.Conv2D(**_lateral_dict)
        self.l_conv2d_c5 = keras.layers.Conv2D(**_lateral_dict)

        # same branch
        self.conv2d_p3 = keras.layers.Conv2D(**_same_dict)
        self.conv2d_p4 = keras.layers.Conv2D(**_same_dict)
        self.conv2d_p5 = keras.layers.Conv2D(**_same_dict)

        # down branch
        self.down_conv2d_p6 = keras.layers.Conv2D(**_down_dict)
        self.down_conv2d_p7 = keras.layers.Conv2D(**_down_dict)

    def call(self, inputs, training=None, mask=None):
        # P3, P4, P5, P6, P7
        # 80, 40, 20, 10, 5
        c3, c4, c5 = inputs[0], inputs[1], inputs[2]

        p5 = self.l_conv2d_c5(c5)
        p5 = self.conv2d_p5(p5)

        # Upsample and Merge
        p4 = self.l_conv2d_c4(c4)
        p4 = keras.layers.Add()(
            [keras.layers.UpSampling2D(2, self.interpolation)(p5), p4]
        )
        p4 = self.conv2d_p4(p4)

        # Upsample and Merge
        p3 = self.l_conv2d_c3(c3)
        p3 = keras.layers.Add()(
            [keras.layers.UpSampling2D(2, self.interpolation)(p4), p3]
        )
        p3 = self.conv2d_p3(p3)

        # down-sampling
        p6 = self.down_conv2d_p6(c5)

        # down-sampling
        p7 = self.down_conv2d_p7(tf.nn.relu(p6))

        return p3, p4, p5, p6, p7

    def get_config(self):
        c = super(FPN, self).get_config()
        c.update(
            {
                'interpolation': self.interpolation
            }
        )
        return c


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

    # P5 : (None, 20, 20, 256)
    P5 = keras.layers.Conv2D(kernel_size=3, **setting, name='P5')(P5)

    # P4 : (None, 40, 40, 256)
    P4 = keras.layers.Conv2D(kernel_size=1, **setting, name='C4_reduced')(C4)
    P4 = keras.layers.Add(name='P4_merge')([P5_upsampled, P4])
    # P4_upsampled : (None, 80, 80, 256)
    P4_upsampled = layers.UpsampleLike(name='P4_upsampled')([P4, C3])

    # P4 : (None, 40, 40, 256)
    P4 = keras.layers.Conv2D(kernel_size=3, **setting, name='P4')(P4)

    # P3 : (None, 80, 80, 256)
    P3 = keras.layers.Conv2D(kernel_size=1, **setting, name='C3_reduced')(C3)
    P3 = keras.layers.Add(name='P3_merge')([P4_upsampled, P3])
    P3 = keras.layers.Conv2D(kernel_size=3, **setting, name='P3')(P3)

    # P6 : (None, 10, 10, 256)
    P6 = keras.layers.Conv2D(**down_setting, name='P6')(C5)
    if n:
        P6 = GroupNormalization(name='P6_gn')(P6)
    P6_relu = keras.layers.Activation('relu', name='P6_relu')(P6)

    # P7 : (None, 5, 5, 256)
    P7 = keras.layers.Conv2D(**down_setting, name='P7')(P6_relu)

    if n:
        P7 = GroupNormalization(name='P7_gn')(P7)
        P5 = GroupNormalization(name='P5_gn')(P5)
        P4 = GroupNormalization(name='P4_gn')(P4)
        P3 = GroupNormalization(name='P3_gn')(P3)

    return [P3, P4, P5, P6, P7]
