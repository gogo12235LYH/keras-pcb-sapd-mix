import tensorflow as tf
import tensorflow.keras as keras
from tensorflow_addons.layers import GroupNormalization
from models.layers import WSConv2D

k_init = tf.initializers.RandomNormal(0.0, 0.01)


class FSN(keras.layers.Layer):
    """ Feature Selection Network V3 """

    def __init__(self, width=256, depth=3, fpn_level=5, ws=0, *args, **kwargs):
        super(FSN, self).__init__(*args, **kwargs)

        self.width = width
        self.depth = depth
        self.fpn_level = fpn_level
        self.ws = ws

        # Network
        self.fsn_blocks = []
        for _ in range(depth):
            if ws:
                self.fsn_blocks.append(
                    WSConv2D(
                        filters=width, kernel_size=3, strides=1, padding='same',
                        kernel_initializer=k_init
                    )
                )
            else:
                self.fsn_blocks.append(
                    keras.layers.Conv2D(
                        filters=width, kernel_size=3, strides=1, padding='same',
                        kernel_initializer=k_init
                    )
                )

            self.fsn_blocks.append(GroupNormalization(groups=16, epsilon=1e-5))
            self.fsn_blocks.append(keras.layers.Activation(tf.nn.relu))

        # Replace Flatten
        self.global_avg = keras.layers.GlobalAvgPool2D()

        # Output
        self.dense = keras.layers.Dense(
            units=fpn_level,
            kernel_initializer=k_init,
            activation='softmax'
        )

    def call(self, inputs, training=None, mask=None):
        # (None, None, None, fpn_level * width)
        x = inputs

        # (None, None, None, width)
        for fsn_layer in self.fsn_blocks:
            x = fsn_layer(x)

        # (None, 1, 1, width)
        x = self.global_avg(x)

        # (None, fpn_level)
        x = self.dense(x)
        return x

    def get_config(self):
        cfg = super(FSN, self).get_config()
        cfg.update(
            {'width': self.width, 'depth': self.depth, 'fpn_level': self.fpn_level, 'ws': self.ws}
        )
        return cfg
