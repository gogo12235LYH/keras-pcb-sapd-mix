import tensorflow as tf
import tensorflow.keras as keras


class FSNLoss(keras.layers.Layer):
    def __init__(self, factor=0.1, *args, **kwargs):
        super(FSNLoss, self).__init__(*args, **kwargs)
        self.factor = factor

    def call(self, inputs, **kwargs):
        loss = keras.losses.sparse_categorical_crossentropy(inputs[0], inputs[1]) * self.factor
        loss = tf.math.reduce_mean(loss)
        self.add_loss(loss)
        self.add_metric(loss, name=self.name)
        return loss

    def get_config(self):
        c = super(FSNLoss, self).get_config()
        c.update({
            'factor': self.factor
        })
        return c
