import tensorflow as tf
import tensorflow.keras as keras
import math


def compute_iou(mode='fciou'):
    @tf.function(jit_compile=True)
    def iou_(y_true, y_pred):
        y_true = tf.maximum(y_true, 0)

        pred_left = y_pred[:, :, 0]
        pred_top = y_pred[:, :, 1]
        pred_right = y_pred[:, :, 2]
        pred_bottom = y_pred[:, :, 3]

        pred_width = pred_left + pred_right
        pred_height = pred_top + pred_bottom

        # (num_pos, )
        target_left = y_true[:, :, 0]
        target_top = y_true[:, :, 1]
        target_right = y_true[:, :, 2]
        target_bottom = y_true[:, :, 3]

        target_width = target_left + target_right
        target_height = target_top + target_bottom

        target_area = (target_left + target_right) * (target_top + target_bottom)
        pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)
        w_intersect = tf.minimum(pred_left, target_left) + tf.minimum(pred_right, target_right)
        h_intersect = tf.minimum(pred_bottom, target_bottom) + tf.minimum(pred_top, target_top)
        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect

        classic_iou = (area_intersect + 1e-7) / (area_union + 1e-7)

        if mode == 'iou':
            # (num_pos, )
            iou_loss = -tf.math.log(classic_iou)

        elif mode == 'giou':
            g_w_intersect = tf.maximum(pred_left, target_left) + tf.maximum(pred_right, target_right)
            g_h_intersect = tf.maximum(pred_bottom, target_bottom) + tf.maximum(pred_top, target_top)
            close_area = g_w_intersect * g_h_intersect

            g_iou = classic_iou - (close_area - area_union) / close_area
            iou_loss = 1 - g_iou

        elif mode == 'ciou':
            pred_atan = tf.atan(pred_width / (pred_height + 1e-9))
            target_atan = tf.atan(target_width / (target_height + 1e-9))
            v = 4.0 * keras.backend.pow((pred_atan - target_atan), 2) / (math.pi ** 2)
            a = v / (1 - classic_iou + v)

            c_iou = classic_iou - 1.0 * a * v
            iou_loss = 1 - c_iou

        elif mode == 'fciou':
            target_center_x = target_width * 0.5
            target_center_y = target_height * 0.5
            pred_center_x = target_left - pred_left + pred_width * 0.5
            pred_center_y = target_top - pred_top + pred_height * 0.5
            intersect_diagonal = (target_center_x - pred_center_x) ** 2 + (target_center_y - pred_center_y) ** 2

            g_w_intersect = tf.maximum(pred_left, target_left) + tf.maximum(pred_right, target_right)
            g_h_intersect = tf.maximum(pred_bottom, target_bottom) + tf.maximum(pred_top, target_top)
            max_diagonal = g_h_intersect ** 2 + g_w_intersect ** 2

            pred_atan = tf.atan(pred_width / (pred_height + 1e-9))
            target_atan = tf.atan(target_width / (target_height + 1e-9))
            v = 4.0 * keras.backend.pow((pred_atan - target_atan), 2) / (math.pi ** 2)
            a = v / (1 - classic_iou + v)

            c_iou = classic_iou - 1.0 * a * v - 1.0 * (intersect_diagonal / max_diagonal)
            iou_loss = 1 - c_iou

        else:
            iou_loss = -tf.math.log(classic_iou)

        # compute the normalizer: the number of positive anchors
        normalizer = tf.maximum(1, tf.reduce_prod(tf.shape(y_true)[0:2]))
        normalizer = tf.cast(normalizer, dtype=tf.float32)
        return tf.reduce_sum(iou_loss) / normalizer

    return iou_


def iou_mask(mode='fciou', factor=1.0):
    def iou_mask_(inputs):
        # y_ture: (Batch, Anchor-points, 4 + 2)
        y_true, y_pred, soft_weight, mask = inputs[0][..., :4], inputs[1], inputs[0][..., 4], inputs[0][..., 5]
        y_true = tf.maximum(y_true, 0)

        pred_left = y_pred[:, :, 0]
        pred_top = y_pred[:, :, 1]
        pred_right = y_pred[:, :, 2]
        pred_bottom = y_pred[:, :, 3]

        pred_width = pred_left + pred_right
        pred_height = pred_top + pred_bottom

        target_left = y_true[:, :, 0]
        target_top = y_true[:, :, 1]
        target_right = y_true[:, :, 2]
        target_bottom = y_true[:, :, 3]

        target_width = target_left + target_right
        target_height = target_top + target_bottom

        target_area = (target_left + target_right) * (target_top + target_bottom)
        masked_target_area = tf.boolean_mask(target_area, mask)
        pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)
        masked_pred_area = tf.boolean_mask(pred_area, mask)
        w_intersect = tf.minimum(pred_left, target_left) + tf.minimum(pred_right, target_right)
        h_intersect = tf.minimum(pred_bottom, target_bottom) + tf.minimum(pred_top, target_top)

        g_w_intersect = tf.maximum(pred_left, target_left) + tf.maximum(pred_right, target_right)
        g_h_intersect = tf.maximum(pred_bottom, target_bottom) + tf.maximum(pred_top, target_top)
        close_area = g_w_intersect * g_h_intersect
        masked_close_area = tf.boolean_mask(close_area, mask)

        area_intersect = w_intersect * h_intersect
        masked_area_intersect = tf.boolean_mask(area_intersect, mask)
        masked_area_union = masked_target_area + masked_pred_area - masked_area_intersect

        masked_soft_weight = tf.boolean_mask(soft_weight, mask)
        # (B, N)
        masked_iou = (masked_area_intersect + 1e-7) / (masked_area_union + 1e-7)

        if mode == 'iou':
            masked_iou_loss = -tf.math.log(masked_iou) * masked_soft_weight

        elif mode == 'giou':
            masked_g_iou = masked_iou - (masked_close_area - masked_area_union) / masked_close_area
            masked_iou_loss = (1 - masked_g_iou) * masked_soft_weight

        elif mode == 'ciou':
            pred_atan = tf.atan(pred_width / (pred_height + 1e-9))
            target_atan = tf.atan(target_width / (target_height + 1e-9))

            masked_pred_atan = tf.boolean_mask(pred_atan, mask)
            masked_target_atan = tf.boolean_mask(target_atan, mask)

            v = 4.0 * keras.backend.pow((masked_pred_atan - masked_target_atan), 2) / (math.pi ** 2)
            a = v / (1 - masked_iou + v)
            c_iou = masked_iou - 1.0 * a * v
            masked_iou_loss = (1 - c_iou) * masked_soft_weight

        elif mode == 'fciou':
            target_center_x = target_width * 0.5
            target_center_y = target_height * 0.5
            pred_center_x = target_left - pred_left + pred_width * 0.5
            pred_center_y = target_top - pred_top + pred_height * 0.5
            intersect_diagonal = (target_center_x - pred_center_x) ** 2 + (target_center_y - pred_center_y) ** 2

            max_diagonal = g_h_intersect ** 2 + g_w_intersect ** 2

            r_masked = tf.boolean_mask((intersect_diagonal / max_diagonal), mask)

            pred_atan = tf.atan(pred_width / (pred_height + 1e-9))
            target_atan = tf.atan(target_width / (target_height + 1e-9))

            masked_pred_atan = tf.boolean_mask(pred_atan, mask)
            masked_target_atan = tf.boolean_mask(target_atan, mask)

            v = 4.0 * keras.backend.pow((masked_pred_atan - masked_target_atan), 2) / (math.pi ** 2)
            a = v / (1 - masked_iou + v)
            c_iou = masked_iou - 1.0 * a * v - 1.0 * r_masked

            masked_iou_loss = (1 - c_iou) * masked_soft_weight
            # masked_iou_loss = (1 - c_iou)

        else:
            masked_iou_loss = -tf.math.log(masked_iou) * masked_soft_weight

        # compute the normalizer: the number of positive locations
        num_pos = tf.reduce_sum(mask * soft_weight)
        normalizer = keras.backend.maximum(1., num_pos)
        return tf.reduce_sum(masked_iou_loss) * factor / normalizer

    return iou_mask_


def iou_mask_v2(mode='fciou', factor=1.0):
    def iou_mask_(inputs):
        y_true, y_pred, soft_weight, mask = inputs[0][..., :4], inputs[1], inputs[0][..., 4], inputs[0][..., 5]
        y_true = tf.maximum(y_true, 0)

        iou_loss = iou_method(soft_weight, y_pred, y_true)

        # mask
        masked_iou_loss = tf.boolean_mask(iou_loss, mask)

        # compute the normalizer: the number of positive locations
        num_pos = tf.reduce_sum(mask * soft_weight)
        normalizer = keras.backend.maximum(1., num_pos)
        return factor * tf.reduce_sum(masked_iou_loss) / normalizer

    @tf.function(jit_compile=True)  # only for tensorflow 2.6, sometimes should use experiment_compile
    def iou_method(soft_weight, y_pred, y_true):
        # pred
        pred_left = y_pred[:, :, 0]
        pred_top = y_pred[:, :, 1]
        pred_right = y_pred[:, :, 2]
        pred_bottom = y_pred[:, :, 3]

        pred_width = pred_left + pred_right
        pred_height = pred_top + pred_bottom
        pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)

        # target
        target_left = y_true[:, :, 0]
        target_top = y_true[:, :, 1]
        target_right = y_true[:, :, 2]
        target_bottom = y_true[:, :, 3]

        target_width = target_left + target_right
        target_height = target_top + target_bottom
        target_area = (target_left + target_right) * (target_top + target_bottom)

        # intersection between target and predict bboxes.
        w_intersect = tf.minimum(pred_left, target_left) + tf.minimum(pred_right, target_right)
        h_intersect = tf.minimum(pred_bottom, target_bottom) + tf.minimum(pred_top, target_top)
        area_intersect = w_intersect * h_intersect

        area_union = target_area + pred_area - area_intersect

        iou = (area_intersect + 1e-7) / (area_union + 1e-7)

        if mode == 'iou':
            iou_loss = -tf.math.log(iou) * soft_weight

        elif mode == 'giou':
            # max close area
            g_w_intersect = tf.maximum(pred_left, target_left) + tf.maximum(pred_right, target_right)
            g_h_intersect = tf.maximum(pred_bottom, target_bottom) + tf.maximum(pred_top, target_top)
            close_area = g_w_intersect * g_h_intersect

            g_iou = iou - (close_area - area_union) / close_area
            iou_loss = (1 - g_iou) * soft_weight

        elif mode == 'ciou':
            # without center fixing
            pred_atan = tf.atan(pred_width / (pred_height + 1e-9))
            target_atan = tf.atan(target_width / (target_height + 1e-9))

            v = 4.0 * keras.backend.pow((pred_atan - target_atan), 2) / (math.pi ** 2)
            a = v / (1 - iou + v)
            c_iou = iou - 1.0 * a * v
            iou_loss = (1 - c_iou) * soft_weight

        elif mode == 'fciou':
            # full version of CIoU
            g_w_intersect = tf.maximum(pred_left, target_left) + tf.maximum(pred_right, target_right)
            g_h_intersect = tf.maximum(pred_bottom, target_bottom) + tf.maximum(pred_top, target_top)
            max_diagonal = g_h_intersect ** 2 + g_w_intersect ** 2

            target_center_x = target_width * 0.5
            target_center_y = target_height * 0.5
            pred_center_x = target_left - pred_left + pred_width * 0.5
            pred_center_y = target_top - pred_top + pred_height * 0.5
            intersect_diagonal = (target_center_x - pred_center_x) ** 2 + (target_center_y - pred_center_y) ** 2

            pred_atan = tf.atan(pred_width / (pred_height + 1e-9))
            target_atan = tf.atan(target_width / (target_height + 1e-9))

            v = 4.0 * keras.backend.pow((pred_atan - target_atan), 2) / (math.pi ** 2)
            a = v / (1 - iou + v)
            c_iou = iou - 1.0 * a * v - 1.0 * (intersect_diagonal / max_diagonal)
            iou_loss = (1 - c_iou) * soft_weight

        else:
            iou_loss = -tf.math.log(iou) * soft_weight
        return iou_loss

    return iou_mask_


class IoULoss(keras.layers.Layer):
    def __init__(self, mode='fciou', factor=1.0, *args, **kwargs):
        super(IoULoss, self).__init__(*args, **kwargs)
        self.mode = mode
        self.factor = factor
        self.fnc = iou_mask_v2(mode, factor)

    def call(self, inputs, **kwargs):
        loss = self.fnc(inputs)
        self.add_loss(loss)
        self.add_metric(loss, name=self.name)
        return loss

    def get_config(self):
        cfg = super(IoULoss, self).get_config()
        cfg.update(
            {'mode': self.mode, 'factor': self.factor}
        )
        return cfg


if __name__ == '__main__':
    temp = IoULoss(mode='fciou', factor=1.0, name='reg_loss')
