import tensorflow as tf
import tensorflow.keras as keras
from tensorflow_addons.layers import GroupNormalization
from models.layers import WSConv2D

k_init = tf.initializers.RandomNormal(0.0, 0.01)


class FSN(keras.Model):
    """ Feature Selection Network V3 """

    def __init__(self, width=256, depth=3, fpn_level=5, ws=0, *args, **kwargs):
        super(FSN, self).__init__(*args, **kwargs)

        # Network
        self.fsn_blocks = keras.Sequential()
        for _ in range(depth):
            if ws:
                self.fsn_blocks.add(
                    WSConv2D(
                        filters=width, kernel_size=3, strides=1, padding='same',
                        kernel_initializer=k_init
                    )
                )
            else:
                self.fsn_blocks.add(
                    keras.layers.Conv2D(
                        filters=width, kernel_size=3, strides=1, padding='same',
                        kernel_initializer=k_init
                    )
                )

            self.fsn_blocks.add(GroupNormalization(groups=16, epsilon=1e-5))
            self.fsn_blocks.add(keras.layers.Activation(tf.nn.relu))

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
        x = self.fsn_blocks(x)

        # (None, 1, 1, width)
        x = self.global_avg(x)

        # (None, fpn_level)
        x = self.dense(x)
        return x

    def get_config(self):
        # TODO: Implement?
        pass
