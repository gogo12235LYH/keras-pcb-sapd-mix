import tensorflow as tf
import tensorflow.keras.backend as k


def focal_loss(alpha=0.25,
               gamma=2.0):
    def focal_loss_(y_true, y_pred):
        """

        Source : https://arxiv.org/abs/1708.02002

        Loss : -1 * alpha * ( 1 - pred) ** gamma * log(pred)
             : alpha_f * focal_w ** gamma * bce(labels, pred)

        :arg

            y_true : ( Batch, Features, num_cls + 1)

            y_pred : ( Batch, Features, num_cls )

        """
        # (B, FS, 1)
        location_state = y_true[:, :, -1]
        # (B, FS, num_cls)
        labels = y_true[:, :, :-1]

        alpha_f = k.ones_like(labels) * alpha
        alpha_f = tf.where(k.equal(labels, 1), alpha_f, 1-alpha_f)

        # focal
        focal_w = tf.where(k.equal(labels, 1), 1-y_pred, y_pred)
        focal_w = alpha_f * focal_w ** gamma
        total_cls_loss = focal_w * k.binary_crossentropy(target=labels,
                                                         output=y_pred)

        normal = tf.where(k.equal(location_state, 1))
        normal = k.cast(k.shape(normal)[0], k.floatx())
        normal = k.maximum(k.cast_to_floatx(1.0), normal)

        return k.sum(total_cls_loss) / normal

    return focal_loss_


def iou_loss():
    def iou_loss_(y_true, y_pred):
        """
        :arg
            y_true : ( Batch, Features, regression )
            y_pred : ( Batch, Features, regression + 2 )
            **note : regression' shape = 4
        :return:
        """
        location_state = y_true[:, :, -1]
        indices = tf.where(k.equal(location_state, 1))

        return tf.cond(tf.size(indices) != 0, lambda: true_fun(y_true, y_pred, indices), lambda: tf.constant(0.0))

    def true_fun(y_true, y_pred, indices):
        y_reg_pred = tf.gather_nd(y_pred, indices)  # (B, 4)
        y_true = tf.gather_nd(y_true, indices)  # (B, 6)
        y_reg_true = y_true[:, :4]  # (B, 4)
        y_cnt_true = y_true[:, 4]  # (B, 1)

        pred_left = y_reg_pred[:, 0]
        pred_top = y_reg_pred[:, 1]
        pred_right = y_reg_pred[:, 2]
        pred_bottom = y_reg_pred[:, 3]

        target_left = y_reg_true[:, 0]
        target_top = y_reg_true[:, 1]
        target_right = y_reg_true[:, 2]
        target_bottom = y_reg_true[:, 3]

        target_area = (target_left + target_right) * (target_top + target_bottom)
        pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)

        width_intersect = tf.minimum(pred_left, target_left) + tf.minimum(pred_right, target_right)
        height_intersect = tf.minimum(pred_top, target_top) + tf.minimum(pred_bottom, target_bottom)
        area_intersect = width_intersect * height_intersect
        area_union = target_area + pred_area - area_intersect

        ious = (area_intersect + 1.0) / (area_union + 1.0)

        # g_width_intersect = tf.maximum(pred_left, target_left) + tf.maximum(pred_right, target_right)
        # g_height_intersect = tf.maximum(pred_top, target_top) + tf.maximum(pred_bottom, target_bottom)
        # ac_union = g_width_intersect * g_height_intersect
        #
        # gious = ious - 1.0 * (ac_union - area_union) / ac_union

        output_loss = -k.log(ious)
        # output_loss = 1.0 - gious
        output_loss = tf.reduce_sum(output_loss * y_cnt_true) / (tf.reduce_sum(y_cnt_true) + 1e-8)
        return output_loss

    return iou_loss_


def bce_center_loss():
    def bce_center_loss_(y_true, y_pred):
        """
        :arg

            y_true : ( Batch, Features, centerness )

            y_pred : ( Batch, Features, centerness + 1 )

            **note : centerness' shape = 1

        :return:
        """
        location_state = y_true[:, :, -1]
        indices = tf.where(k.equal(location_state, 1))

        return tf.cond(tf.size(indices) != 0, lambda: true_fun(y_true, y_pred, indices), lambda: tf.constant(0.0))

    def true_fun(y_true, y_pred, indices):
        y_cnt_pred = tf.gather_nd(y_pred, indices)
        y_true = tf.gather_nd(y_true, indices)
        y_cnt_true = y_true[:, 0:1]

        output_loss = k.switch(tf.size(y_cnt_true > 0),
                               k.binary_crossentropy(target=y_cnt_true, output=y_cnt_pred),
                               tf.constant(0.0))
        return output_loss

    return bce_center_loss_

