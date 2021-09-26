import tensorflow as tf
import tensorflow.keras as keras
import math
import config


def q_focal(alpha=0, beta=2.0, cutoff=0.0):
    def q_focal_(y_true, y_pred):
        """
            Quality Focal Loss: BCE(y_true - y_pred) * ABS(y_true - y_pred) ** beta
            Y_pred  : Sigmoid output
            Target  : (Batch, Positive Anchor-points, classes)
            Predict : (Batch, Positive Anchor-points, classes)
        """

        """ Positive sample with alpha and Negative sample with 1 - alpha """
        alpha_factor = tf.ones_like(y_true) * alpha
        alpha_factor = tf.where(tf.greater(y_true, cutoff), alpha_factor, 1 - alpha_factor)
        alpha_factor = alpha_factor if alpha > 0. else 1.

        """ Modulating factor: ABS(Y_true - Y_pred) ** beta """
        modulating_factor = tf.where(tf.greater(y_true, cutoff), y_true - y_pred, y_pred)
        modulating_factor = alpha_factor * tf.math.abs(modulating_factor) ** beta

        """ Q-Focal loss """
        q_focal_loss = alpha_factor * modulating_factor * tf.keras.backend.binary_crossentropy(y_true, y_pred)

        """ compute the normalizer: the number of positive locations """
        normalizer = tf.cast(tf.shape(y_pred)[1], tf.float32)
        normalizer = tf.maximum(1.0, normalizer)
        return tf.reduce_sum(q_focal_loss) / normalizer

    return q_focal_


def q_focal_mask(alpha=0, beta=2.0, cutoff=0.0):
    def q_focal_mask_(inputs):
        """
            Quality Focal Loss: BCE(y_true - y_pred) * ABS(y_true - y_pred) ** beta
            Y_pred  : Sigmoid output
            Cls_Target  : (Batch, Anchor-points, classes + 2)
            Cls_Predict : (Batch, Anchor-points, classes)
        """

        cls_target, soft_weight, mask = inputs[0][..., :-2], inputs[0][..., -2], inputs[0][..., -1]
        cls_pred = inputs[1]

        """ Positive sample with alpha and Negative sample with 1 - alpha """
        alpha_factor = keras.backend.ones_like(cls_target) * alpha
        alpha_factor = tf.where(tf.greater(cls_target, cutoff), alpha_factor, 1 - alpha_factor)
        alpha_factor = alpha_factor if alpha > 0. else 1.

        """ Modulating factor: ABS(Y_true - Y_pred) ** beta """
        modulating_factor = tf.where(tf.greater(cls_target, cutoff), cls_target - cls_pred, cls_pred)
        modulating_factor = alpha_factor * tf.math.abs(modulating_factor) ** beta

        """ Q-Focal loss """
        q_focal_loss = modulating_factor * tf.keras.backend.binary_crossentropy(cls_target, cls_pred)

        """ Soft weights """
        soft_weight = tf.expand_dims(soft_weight, axis=-1)
        cls_loss = soft_weight * q_focal_loss

        """ compute the normalizer: the number of positive locations """
        # SUM(Soft weight)
        num_pos = tf.reduce_sum(mask * soft_weight[..., 0])
        normalizer = tf.maximum(1.0, tf.cast(num_pos, dtype=tf.float32))
        return tf.reduce_sum(cls_loss) / normalizer

    return q_focal_mask_


def focal(alpha=0.25, gamma=2.0, cutoff=0.0):
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


def iou(mode=config.IOU_LOSS):
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

        g_w_intersect = tf.maximum(pred_left, target_left) + tf.maximum(pred_right, target_right)
        g_h_intersect = tf.maximum(pred_bottom, target_bottom) + tf.maximum(pred_top, target_top)
        close_area = g_w_intersect * g_h_intersect

        classic_iou = (area_intersect + 1e-7) / (area_union + 1e-7)

        if mode == 'iou':
            # (num_pos, )
            iou_loss = -tf.math.log(classic_iou)

        elif mode == 'giou':
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

            # delta_x = (target_width - pred_width) * 0.5 - (target_left - pred_left)
            # delta_y = (target_height - pred_height) * 0.5 - (target_top - pred_top)
            # intersect_diagonal = delta_x ** 2 + delta_y ** 2
            max_diagonal = g_h_intersect ** 2 + g_w_intersect ** 2

            r = intersect_diagonal / max_diagonal

            pred_atan = tf.atan(pred_width / (pred_height + 1e-9))
            target_atan = tf.atan(target_width / (target_height + 1e-9))
            v = 4.0 * keras.backend.pow((pred_atan - target_atan), 2) / (math.pi ** 2)
            a = v / (1 - classic_iou + v)

            c_iou = classic_iou - 1.0 * a * v - 1.0 * r
            iou_loss = 1 - c_iou

        else:
            iou_loss = -tf.math.log(classic_iou)

        # compute the normalizer: the number of positive anchors
        normalizer = tf.maximum(1, tf.reduce_prod(tf.shape(y_true)[0:2]))
        normalizer = tf.cast(normalizer, dtype=tf.float32)
        return tf.reduce_sum(iou_loss) / normalizer

    return iou_


def iou_mask(mode=config.IOU_LOSS, factor=1.0):
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


def _loss():
    def fake_loss_(y_true, y_pred):
        return y_pred
    return fake_loss_


def total_loss(cls='cls_loss', reg='reg_loss', fsn_loss='feature_select_loss'):
    return {
        cls: _loss(),
        reg: _loss(),
        fsn_loss: _loss()
    }
