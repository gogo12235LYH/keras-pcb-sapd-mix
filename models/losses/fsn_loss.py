import tensorflow.keras as keras


class FSNLoss(keras.layers.Layer):
    def __init__(self, factor=0.1, *args, **kwargs):
        super(FSNLoss, self).__init__(*args, **kwargs)
        self.factor = factor

    def call(self, inputs, **kwargs):
        return keras.losses.sparse_categorical_crossentropy(inputs[0], inputs[1]) * self.factor

    def compute_output_shape(self, input_shape):
        return [1, ]

    def get_config(self):
        c = super(FSNLoss, self).get_config()
        c.update(
            {
                'factor': self.factor
            }
        )
        return c
