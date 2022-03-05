import tensorflow as tf
import tensorflow.keras as keras


def compute_focal(alpha=0.25, gamma=2.0, cutoff=0.0):
    @tf.function(jit_compile=True)
    def focal_(y_true, y_pred):
        """
            Focal Loss(Pos): CE(y_ture, y_pred) * alpha * (1 - pred) ** beta
            Focal Loss(Neg): CE(y_ture, y_pred) * (1 - alpha) * pred ** beta
        """

        """ Positive sample: alpha, Negative sample: 1 - alpha """
        alpha_factor = tf.ones_like(y_true) * alpha
        alpha_factor = tf.where(tf.greater(y_true, cutoff), alpha_factor, 1 - alpha_factor)

        """ Positive sample: 1 - pred, Negative sample: pred """
        focal_weight = tf.where(tf.greater(y_true, cutoff), 1 - y_pred, y_pred)
        focal_weight = alpha_factor * focal_weight ** gamma

        """ Focal loss """
        classification_loss = focal_weight * keras.backend.binary_crossentropy(y_true, y_pred)

        """ Number of samples """
        normalizer = tf.cast(tf.shape(y_pred)[1], tf.float32)
        normalizer = tf.maximum(1.0, normalizer)
        return tf.reduce_sum(classification_loss) / normalizer

    return focal_


def focal_mask(alpha=0.25, gamma=2.0, cutoff=0.0):
    @tf.function(jit_compile=True)
    def focal_mask_(inputs):
        """
            Focal Loss(Pos): CE(y_ture, y_pred) * alpha * (1 - pred) ** beta
            Focal Loss(Neg): CE(y_ture, y_pred) * (1 - alpha) * pred ** beta
        """

        # y_ture: (Batch, Anchor-points, classes + 2)
        y_true, soft_weight, mask = inputs[0][..., :-2], inputs[0][..., -2], inputs[0][..., -1]

        # y_pred: (Batch, Anchor-points, classes)
        y_pred = inputs[1]

        alpha_factor = keras.backend.ones_like(y_true) * alpha
        alpha_factor = tf.where(tf.greater(y_true, cutoff), alpha_factor, 1 - alpha_factor)

        focal_weight = tf.where(tf.greater(y_true, cutoff), 1 - y_pred, y_pred)
        focal_weight = alpha_factor * focal_weight ** gamma

        soft_weight = tf.expand_dims(soft_weight, axis=-1)
        cls_loss = focal_weight * soft_weight * tf.keras.backend.binary_crossentropy(y_true, y_pred)

        # compute the normalizer: the number of positive locations
        num_pos = tf.reduce_sum(mask * soft_weight[..., 0])
        normalizer = tf.maximum(1.0, tf.cast(num_pos, dtype=tf.float32))
        return tf.reduce_sum(cls_loss) / normalizer

    return focal_mask_


class FocalLoss(keras.layers.Layer):
    def __init__(self, alpha=0.25, gamma=2.0, *args, **kwargs):
        super(FocalLoss, self).__init__(dtype='float32', *args, **kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.fnc = focal_mask(alpha=alpha, gamma=gamma)

    def call(self, inputs, **kwargs):
        loss = self.fnc(inputs)
        self.add_loss(loss)
        self.add_metric(loss, name=self.name)
        return loss

    def get_config(self):
        cfg = super(FocalLoss, self).get_config()
        cfg.update(
            {'alpha': self.alpha, 'gamma': self.gamma}
        )
        return cfg
