import tensorflow.keras as keras
import keras_resnet.models
import tensorflow as tf
from utils.util_graph import trim_zero_padding_boxes, normalize_boxes, shrink_and_normalize_boxes
from models.losses_add import q_focal
from models.neck.fpn import create_pyramid_features_v2
from models.layers import Locations2, RegressionBoxes2, ClipBoxes2, FilterDetections2
from models.losses import FocalLoss, IoULoss, FSNLoss, compute_iou, compute_focal
import config

NUM_CLS = config.NUM_CLS
STRIDES = (8, 16, 32, 64, 128)
SR = config.SHRINK_RATIO


class FeatureSelectInput(keras.layers.Layer):
    def __init__(self, strides=STRIDES, pool_size=7, **kwargs):
        self.strides = strides
        self.pool_size = pool_size
        super(FeatureSelectInput, self).__init__(**kwargs)

    @tf.function
    def call(self, inputs, **kwargs):
        # (Batch, Max_Bboxes_count, 4)
        batch_gt_boxes = inputs[0][..., :4]

        # (P3's shape, P4's shape, P5's shape, P6's shape, P7's shape)
        list_batch_feature_maps = inputs[1: 1 + len(self.strides)]

        # If Batch size = 2
        batch_size = tf.shape(batch_gt_boxes)[0]
        # Default : 100
        max_gt_boxes_count = tf.shape(batch_gt_boxes)[1]

        # (Batch, MaxBboxes_count)
        gt_boxes_ids = tf.tile(tf.expand_dims(tf.range(batch_size), axis=-1), (1, max_gt_boxes_count))
        # (Batch * Max_Bboxes_count, )
        gt_boxes_ids = tf.reshape(gt_boxes_ids, (-1,))

        # (Batch * Max_Bboxes_count, 4)
        batch_gt_boxes = tf.reshape(batch_gt_boxes, (-1, tf.shape(batch_gt_boxes)[-1]))

        # batch_gt_true_boxes: (total_True_Label_count, 4)
        # non_zeros_mask:  (Batch * Max_Bboxes_count, )
        batch_gt_true_boxes, non_zeros_mask = trim_zero_padding_boxes(batch_gt_boxes)

        # (batch * max_bboxes_count, ) -> (total_True_Label_count, )
        gt_boxes_ids = tf.boolean_mask(gt_boxes_ids, non_zeros_mask)

        rois_from_feature_maps = []
        # list: (Fpn_level, Batch, fmap_shape, fmap_shape, 256)
        for i, batch_feature_map in enumerate(list_batch_feature_maps):
            # batch_feature_map: (Batch, fmap_shape, fmap_shape, 256)
            # [8, 16, 32, 64, 128]
            stride = tf.constant(self.strides[i], dtype=tf.float32)
            feature_map_height = tf.cast(tf.shape(batch_feature_map)[1], dtype=tf.float32)
            feature_map_width = tf.cast(tf.shape(batch_feature_map)[2], dtype=tf.float32)

            # (total_True_label_count, 4)
            normalized_gt_boxes = normalize_boxes(boxes=batch_gt_true_boxes,
                                                  width=feature_map_width,
                                                  height=feature_map_height,
                                                  stride=stride
                                                  )

            # (batch_size * true_label_count, pool_size, pool_size, feature_map_channel)
            roi = tf.image.crop_and_resize(image=batch_feature_map,
                                           boxes=normalized_gt_boxes,
                                           box_indices=gt_boxes_ids,
                                           crop_size=(self.pool_size, self.pool_size)
                                           )

            # rois_from_fmaps: (Fpn_level, ) - (total_true_label_count, pool_size, pool_size, 256)
            rois_from_feature_maps.append(roi)

        # (total_True_label_count, pool_size, pool_size, feature_map_channel * 5)
        rois = tf.concat(rois_from_feature_maps, axis=-1)
        return rois, gt_boxes_ids

    def get_config(self):
        c = super(FeatureSelectInput, self).get_config()
        c.update({
            "strides": self.strides,
            "pool_size": self.pool_size,
        })
        return c


@tf.function
def _create_reg_positive_sample(bboxes, x1, y1, x2, y2, stride):
    shift_xx = (tf.cast(tf.range(x1, x2), dtype=tf.float32) + 0.5) * stride
    shift_yy = (tf.cast(tf.range(y1, y2), dtype=tf.float32) + 0.5) * stride
    shift_xx, shift_yy = tf.meshgrid(shift_xx, shift_yy)
    shifts = tf.stack((shift_xx, shift_yy), axis=-1)

    l = tf.maximum(shifts[..., 0] - bboxes[0], 0)
    t = tf.maximum(shifts[..., 1] - bboxes[1], 0)
    r = tf.maximum(bboxes[2] - shifts[..., 0], 0)
    b = tf.maximum(bboxes[3] - shifts[..., 1], 0)

    reg_target = tf.stack((l, t, r, b), axis=-1) / 4.0 / stride
    anchor_pots = tf.minimum(l, r) * tf.minimum(t, b) / tf.maximum(l, r) / tf.maximum(t, b)
    area = (l + r) * (t + b)

    return reg_target, anchor_pots, area


def _build_map_function_feature_select_target(
        cls_pred,
        reg_pred,
        feature_shapes,
        strides,
        gt_boxes,
        shrink_ratio=SR
):
    """
    :param cls_pred: Header model 預測類別.
    :param reg_pred: Header model 預測標記框回歸.
    :param feature_shapes: FPN(P3, P4, P5, P6, P7), 各層feature map 大小.
    :param strides: (8, 16, 32, 64, 128).
    :param gt_boxes: (max_gt_boxes_count, 5).
    :param shrink_ratio: 面積收縮率.
    :return:
    """
    # (Max_Bboxes_count, 1)
    gt_labels = tf.cast(gt_boxes[:, 4], tf.int32)

    # (Max_Bboxes_count, 4)
    gt_boxes = gt_boxes[:, :4]
    max_gt_boxes = tf.shape(gt_boxes)[0]

    focal_loss = q_focal() if config.USING_QFL else compute_focal()
    # focal_loss = focal()
    iou_loss = compute_iou(mode=config.IOU_LOSS)

    # (True_Label_count, 4); (True_Label_count, )
    gt_boxes, non_zeros = trim_zero_padding_boxes(gt_boxes)
    num_gt_boxes = tf.shape(gt_boxes)[0]

    # (True_Label_count, 1)
    gt_labels = tf.boolean_mask(gt_labels, non_zeros)

    level_losses = []

    # @tf.function
    def _get_iou_score(target, pred):
        # t_l = target[:, 0]
        # t_t = target[:, 1]
        # t_r = target[:, 2]
        # t_b = target[:, 3]
        #
        # p_l = pred[:, 0]
        # p_t = pred[:, 1]
        # p_r = pred[:, 2]
        # p_b = pred[:, 3]
        #
        # t_area = (t_l + t_r) * (t_t + t_b)
        # p_area = (p_l + p_r) * (p_t + p_b)
        #
        # i_width = tf.minimum(t_l, p_l) + tf.minimum(t_r, p_r)
        # i_height = tf.minimum(t_t, p_t) + tf.minimum(t_b, p_b)
        # i_area = i_width * i_height

        t_area = (target[..., 0] + target[..., 2]) * (target[..., 1] + target[..., 3])
        p_area = (pred[..., 0] + pred[..., 2]) * (pred[..., 1] + pred[..., 3])

        i_width = tf.minimum(target[..., 0], pred[..., 0]) + tf.minimum(target[..., 2], pred[..., 2])
        i_height = tf.minimum(target[..., 1], pred[..., 1]) + tf.minimum(target[..., 3], pred[..., 3])
        i_area = i_width * i_height

        u_area = t_area + p_area - i_area + 1e-7
        return i_area / u_area

    # Feature map area: [6400, 1600, 400, 100, 25]
    fa = tf.reduce_prod(feature_shapes, axis=-1)

    for level_id in range(len(strides)):
        stride = strides[level_id]

        # Feature map height and width: [80, 40, 20, 10, 5]
        fh = feature_shapes[level_id][0]
        fw = feature_shapes[level_id][1]

        start_idx = tf.reduce_sum(fa[:level_id])
        end_idx = start_idx + fh * fw
        cls_pred_i = tf.reshape(cls_pred[start_idx:end_idx], (fh, fw, tf.shape(cls_pred)[-1]))
        reg_pred_i = tf.reshape(reg_pred[start_idx:end_idx], (fh, fw, tf.shape(reg_pred)[-1]))

        # (True_Label_count, )
        x1, y1, x2, y2 = shrink_and_normalize_boxes(gt_boxes, fw, fh, stride, shrink_ratio=shrink_ratio)

        def compute_gt_box_loss(args):
            x1_ = args[0]
            y1_ = args[1]
            x2_ = args[2]
            y2_ = args[3]
            gt_box = args[4]
            gt_label = args[5]

            def true_function():
                """ Target """

                """ Feature map from Classification Subnet """
                locs_cls_pred_i = cls_pred_i[y1_:y2_, x1_:x2_, :]
                locs_cls_pred_i = tf.reshape(locs_cls_pred_i, (-1, tf.shape(locs_cls_pred_i)[-1]))

                """ Feature map from Regression Subnet """
                locs_reg_pred_i = reg_pred_i[y1_:y2_, x1_:x2_, :]
                locs_reg_pred_i = tf.reshape(locs_reg_pred_i, (-1, tf.shape(locs_reg_pred_i)[-1]))

                """ Creating Positive sample from Regression Subnet """
                locs_reg_true_i = _create_reg_positive_sample(gt_box, x1_, y1_, x2_, y2_, stride)[0]
                locs_reg_true_i = tf.reshape(locs_reg_true_i, (-1, 4))

                """ Creating Positive sample from Classification Subnet """
                locs_cls_true_i = tf.zeros_like(locs_cls_pred_i)

                if config.USING_QFL:
                    # gt_label_col = _get_iou_score2(shifts, gt_box, locs_reg_pred_i, stride)
                    gt_label_col = _get_iou_score(locs_reg_true_i, locs_reg_pred_i)
                    gt_label_col = tf.expand_dims(gt_label_col, axis=-1)

                else:
                    gt_label_col = tf.ones_like(locs_cls_true_i[:, 0:1])

                locs_cls_true_i = tf.concat([locs_cls_true_i[:, :gt_label],
                                             gt_label_col,
                                             locs_cls_true_i[:, gt_label + 1:],
                                             ], axis=-1)

                """ Loss for FSN """
                loss_cls = focal_loss(tf.expand_dims(locs_cls_true_i, axis=0), tf.expand_dims(locs_cls_pred_i, axis=0))
                loss_reg = iou_loss(tf.expand_dims(locs_reg_true_i, axis=0), tf.expand_dims(locs_reg_pred_i, axis=0))

                return loss_cls + loss_reg

            def false_function():
                box_loss = tf.constant(1e7, dtype=tf.float32)
                return box_loss

            level_box_loss = tf.cond(
                tf.equal(tf.cast(x1_, tf.int32), tf.cast(x2_, tf.int32)) |
                tf.equal(tf.cast(y1_, tf.int32), tf.cast(y2_, tf.int32)),
                false_function,
                true_function
            )
            return level_box_loss

        # (True_Label_count, )
        level_loss = tf.map_fn(
            compute_gt_box_loss,
            elems=[x1, y1, x2, y2, gt_boxes, gt_labels],
            fn_output_signature=tf.float32,
        )
        level_losses.append(level_loss)

    # list: (True_Label_count, ) - (5, ) -> (True_Label_count, 5)
    losses = tf.stack(level_losses, axis=-1)

    # (True_Label_count, 1)
    gt_box_levels = tf.argmin(losses, axis=-1, output_type=tf.int32)
    padding_gt_box_levels = tf.ones((max_gt_boxes - num_gt_boxes), dtype=tf.int32) * -1

    # (Max_Bboxes_count, 1)
    gt_box_levels = tf.concat([gt_box_levels, padding_gt_box_levels], axis=0)
    return gt_box_levels


class FeatureSelectTarget(keras.layers.Layer):
    def __init__(self, strides=STRIDES, shrink_ratio=SR, **kwargs):
        self.strides = strides
        self.shrink_ratio = shrink_ratio
        super(FeatureSelectTarget, self).__init__(**kwargs)

    # @tf.function
    def call(self, inputs, **kwargs):
        # (B, sum(F_maps), num_cls)
        batch_cls_pred = inputs[0]
        # (B, sum(F_maps), 4)
        batch_reg_pred = inputs[1]
        # (B, 5, 2)
        feature_maps_shape = inputs[2][0]
        # (B, Max_Bboxes_count, 5)
        batch_gt_boxes = inputs[3]

        def _build_feature_select_target(args):
            cls_pred = args[0]
            reg_pred = args[1]
            gt_boxes = args[2]

            return _build_map_function_feature_select_target(
                cls_pred=cls_pred,
                reg_pred=reg_pred,
                feature_shapes=feature_maps_shape,
                strides=self.strides,
                gt_boxes=gt_boxes,
                shrink_ratio=self.shrink_ratio
            )

        # (Batch, Max_Bboxes_count)
        batch_boxes_level = tf.map_fn(
            _build_feature_select_target,
            elems=[batch_cls_pred, batch_reg_pred, batch_gt_boxes],
            fn_output_signature=tf.int32
        )

        # (Batch * Max_Bboxes_count, )
        batch_boxes_level = tf.reshape(batch_boxes_level, (-1,))
        # (Batch * Max_Bboxes_count, )
        mask = tf.not_equal(batch_boxes_level, -1)
        # (total_true_label_count, )
        batch_boxes_level = tf.boolean_mask(batch_boxes_level, mask)
        return batch_boxes_level

    def get_config(self):
        c = super(FeatureSelectTarget, self).get_config()
        c.update({
            'strides': self.strides,
            'shrink_ratio': self.shrink_ratio
        })
        return c


# 2020-10-19, 選取最高3個權重(限Soft Weight)
def _build_map_function_top_soft_weight(soft_weight, top_k=3):
    assert 6 > top_k > 0

    def build_map_function_top(args):
        instance_weight = args[0]
        zeros_weight = tf.zeros_like(instance_weight)

        values, ids = tf.math.top_k(instance_weight, k=top_k)
        min_weight = tf.math.reduce_min(values)

        return tf.where(
            tf.greater_equal(instance_weight, min_weight),
            instance_weight,
            zeros_weight,
        )

    return tf.map_fn(
        build_map_function_top,
        elems=[soft_weight],
        fn_output_signature=tf.float32
    )


# refactor - 20211105
@tf.function(jit_compile=True)
def _build_map_function_top_soft_weight_test(soft_weight, top_k=3):
    # soft_weight: (None, 5)
    # topk_value: (None, top_k)
    topk_min = tf.math.top_k(soft_weight, k=top_k + 1)[0][..., -1, None]

    boolean_mask = soft_weight > topk_min

    return tf.where(
        boolean_mask,
        soft_weight,
        0.
    )


# 2020-10-19, 選取最高3個權重(限Soft Weight)
class FeatureSelectWeight_V1(keras.layers.Layer):
    def __init__(self,
                 max_gt_boxes_count=100,
                 soft=True,
                 batch_size=4,
                 **kwargs):

        self.max_gt_boxes_count = max_gt_boxes_count
        self.soft = soft
        self.batch_size = batch_size
        super(FeatureSelectWeight_V1, self).__init__(**kwargs)

    # @tf.function
    def call(self, inputs, **kwargs):
        # If Batch size = 1
        # max_gt_boxes_count = 100
        # batch_True_Label_count = 6

        if self.soft:
            # (sum(Batch_True_Label_count), select_weight_pred) = (1 * 6, 5)
            # gt_boxes_select_weight = inputs[0]
            gt_boxes_select_weight = _build_map_function_top_soft_weight(inputs[0], top_k=3)

        else:
            # (sum(Batch_True_Label_count), select_weight_pred) = (1 * 6, 5)
            gt_boxes_select_weight = tf.one_hot(inputs[0], 5)
            # gt_boxes_select_weight = tf.ones_like(tf.shape(inputs[0]))

        # (sum(Batch_True_Label_count), )
        # (1 * 6, ) : [0, 0, 0, 0, 0, 0, ]
        gt_boxes_batch_ids = inputs[1]

        # (Batch_size, 1) -> (Batch_size, ) = (1, ) : [6, ]
        batch_true_label_gt_boxes_count = inputs[2][..., 0]

        batch_select_weight = []
        for i in range(self.batch_size):
            # (Batch_True_Label_count, 5) = (6, 5)
            batch_item_select_weight = tf.boolean_mask(gt_boxes_select_weight, tf.equal(gt_boxes_batch_ids, i))

            # [[0, Max_Bboxes_count - True_Label_count], [0, 0]] = [[0, 94], [0, 0]]
            pad_top_bot = tf.stack([tf.constant(0),
                                    self.max_gt_boxes_count - batch_true_label_gt_boxes_count[i]
                                    ], axis=0)
            pad = tf.stack([pad_top_bot, tf.constant([0, 0])], axis=0)

            # (Max_Bboxes_count, 5) = (100, 5)
            batch_select_weight.append(tf.pad(batch_item_select_weight, pad, constant_values=-1))

        # (Batch, Max_Bboxes_count, 5) = (1, 100, 5)
        batch_select_weight = tf.stack(batch_select_weight, axis=0)
        return batch_select_weight

    def get_config(self):
        c = super(FeatureSelectWeight_V1, self).get_config()
        c.update({
            'max_gt_boxes_count': self.max_gt_boxes_count,
            'soft': self.soft
        })
        return c


class FeatureSelectWeight_V2(keras.layers.Layer):
    def __init__(self,
                 max_gt_boxes_count=100,
                 soft=True,
                 **kwargs):

        self.max_gt_boxes_count = max_gt_boxes_count
        self.soft = soft

        super(FeatureSelectWeight_V2, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        # If Batch size = 1
        # max_gt_boxes_count = 100
        # batch_True_Label_count = 6

        if self.soft:
            # (sum(Batch_True_Label_count), select_weight_pred) = (1 * 6, 5)
            gt_boxes_select_weight = inputs[0]
            if 1 < config.FSN_TOP_K < 5:
                gt_boxes_select_weight = _build_map_function_top_soft_weight_test(
                    gt_boxes_select_weight, top_k=config.FSN_TOP_K
                )

        else:
            # (sum(Batch_True_Label_count), select_weight_pred) = (1 * 6, 5)
            if config.TOP1_TRAIN:
                gt_boxes_select_weight = tf.one_hot(inputs[0], 5)
            else:
                gt_boxes_select_weight = tf.ones_like(tf.shape(inputs[0]))
                gt_boxes_select_weight = tf.expand_dims(gt_boxes_select_weight, axis=-1)
                gt_boxes_select_weight = tf.cast(tf.tile(gt_boxes_select_weight, (1, 5)), tf.float32)

        # (sum(Batch_True_Label_count), )
        # (1 * 6, ) : [0, 0, 0, 0, 0, 0, ]
        gt_boxes_batch_ids = inputs[1]

        # (Batch_size, 1) -> (Batch_size, ) = (1, ) : [6, ]
        batch_true_label_gt_boxes_count = inputs[2][..., 0]

        batch_select_weight = _batch_select_weight(
            batch_true_label_gt_boxes_count=batch_true_label_gt_boxes_count,
            max_gt_boxes_count=self.max_gt_boxes_count,
            gt_boxes_select_weight=gt_boxes_select_weight,
            gt_boxes_batch_ids=gt_boxes_batch_ids
        )

        return batch_select_weight

    def get_config(self):
        c = super(FeatureSelectWeight_V2, self).get_config()
        c.update({
            'max_gt_boxes_count': self.max_gt_boxes_count,
            'soft': self.soft
        })
        return c


# 2020-1120, Replace function from 'for' to 'tf.map_fn'
# @tf.function
def _batch_select_weight(batch_true_label_gt_boxes_count=None,
                         max_gt_boxes_count=None,
                         gt_boxes_select_weight=None,
                         gt_boxes_batch_ids=None
                         ):
    batch_size = tf.shape(batch_true_label_gt_boxes_count)[0]
    batch_ids_ = tf.range(start=0, limit=batch_size)
    batch_ids_ = tf.reshape(batch_ids_, [-1, 1])

    def _build_map_fn(arg):
        true_label_gt_boxes_count, batch_ids = arg[0], arg[1]

        batch_item_select_weight = tf.boolean_mask(
            gt_boxes_select_weight,
            tf.equal(gt_boxes_batch_ids, batch_ids)
        )
        pad_top_bot = tf.stack([tf.constant(0),
                                max_gt_boxes_count - true_label_gt_boxes_count
                                ], axis=0)
        pad = tf.stack([pad_top_bot, tf.constant([0, 0])], axis=0)

        # (Max_Bboxes_count, 5)
        return tf.pad(batch_item_select_weight, pad, constant_values=-1)

    # (Batch, Max_Bboxes_count, 5)
    return tf.map_fn(
        fn=_build_map_fn,
        elems=[batch_true_label_gt_boxes_count, batch_ids_],
        fn_output_signature=tf.float32,
    )


def _build_map_function_module_target(
        gt_boxes,
        feature_select_weight,
        feature_maps_shape,
        shrink_ratio=SR
):
    num_cls = NUM_CLS
    strides_ = (8, 16, 32, 64, 128)

    gt_labels = tf.cast(gt_boxes[:, 4], tf.int32)
    gt_boxes = gt_boxes[:, :4]
    gt_boxes, non_zeros = trim_zero_padding_boxes(gt_boxes)
    gt_labels = tf.boolean_mask(gt_labels, non_zeros)
    meta_select_weight = tf.boolean_mask(feature_select_weight, non_zeros)

    def true_function():
        cls_target_ = tf.zeros((0, num_cls + 1 + 1), dtype=tf.float32)
        regr_target_ = tf.zeros((0, 4 + 1 + 1), dtype=tf.float32)

        for level_id in range(len(strides_)):
            # (objects, )
            level_meta_select_weight = meta_select_weight[:, level_id]

            stride = strides_[level_id]

            fh = feature_maps_shape[level_id][0]
            fw = feature_maps_shape[level_id][1]

            pos_x1, pos_y1, pos_x2, pos_y2 = shrink_and_normalize_boxes(gt_boxes, fw, fh, stride, shrink_ratio)

            def build_map_function_target(args):
                pos_x1_ = args[0]
                pos_y1_ = args[1]
                pos_x2_ = args[2]
                pos_y2_ = args[3]
                gt_box = args[4]
                gt_label = args[5]
                level_box_meta_select_weight = args[6]

                """ Create Negative sample """
                neg_top_bot = tf.stack((pos_y1_, fh - pos_y2_), axis=0)
                neg_lef_rit = tf.stack((pos_x1_, fw - pos_x2_), axis=0)
                neg_pad = tf.stack([neg_top_bot, neg_lef_rit], axis=0)

                """ Regression Target: create positive sample """
                # pos_shift_xx = (tf.cast(tf.range(pos_x1_, pos_x2_), dtype=tf.float32) + 0.5) * stride
                # pos_shift_yy = (tf.cast(tf.range(pos_y1_, pos_y2_), dtype=tf.float32) + 0.5) * stride
                # pos_shift_xx, pos_shift_yy = tf.meshgrid(pos_shift_xx, pos_shift_yy)
                # pos_shifts = tf.stack((pos_shift_xx, pos_shift_yy), axis=-1)
                # dl = tf.maximum(pos_shifts[:, :, 0] - gt_box[0], 0)
                # dt = tf.maximum(pos_shifts[:, :, 1] - gt_box[1], 0)
                # dr = tf.maximum(gt_box[2] - pos_shifts[:, :, 0], 0)
                # db = tf.maximum(gt_box[3] - pos_shifts[:, :, 1], 0)
                # deltas = tf.stack((dl, dt, dr, db), axis=-1)
                # level_box_regr_pos_target = deltas / 4.0 / stride
                # level_pos_box_ap_weight= tf.minimum(dl, dr) * tf.minimum(dt, db) / tf.maximum(dl, dr) / tf.maximum(dt,
                #                                                                                                    db)
                level_box_regr_pos_target, level_pos_box_ap_weight, level_box_pos_area = _create_reg_positive_sample(
                    gt_box, pos_x1_, pos_y1_, pos_x2_, pos_y2_, stride
                )
                level_pos_box_soft_weight = level_pos_box_ap_weight * level_box_meta_select_weight
                # level_pos_box_soft_weight = (1 - level_pos_box_ap_weight) * level_box_meta_select_weight  # ?

                """ Classification Target: create positive sample """
                level_pos_box_cls_target = tf.zeros((pos_y2_ - pos_y1_, pos_x2_ - pos_x1_, num_cls), dtype=tf.float32)
                level_pos_box_gt_label_col = tf.ones((pos_y2_ - pos_y1_, pos_x2_ - pos_x1_, 1), dtype=tf.float32)
                level_pos_box_cls_target = tf.concat((level_pos_box_cls_target[..., :gt_label],
                                                      level_pos_box_gt_label_col,
                                                      level_pos_box_cls_target[..., gt_label + 1:]), axis=-1)

                """ Padding Classification Target's negative sample """
                level_box_cls_target = tf.pad(level_pos_box_cls_target,
                                              tf.concat((neg_pad, tf.constant([[0, 0]])), axis=0))

                """ Padding Soft Anchor's negative sample """
                level_box_soft_weight = tf.pad(level_pos_box_soft_weight, neg_pad, constant_values=1)

                """ Creating Positive Sample locations and padding it's negative sample """
                level_pos_box_regr_mask = tf.ones((pos_y2_ - pos_y1_, pos_x2_ - pos_x1_))
                level_box_regr_mask = tf.pad(level_pos_box_regr_mask, neg_pad)

                """ Padding Regression Target's negative sample """
                level_box_regr_target = tf.pad(level_box_regr_pos_target,
                                               tf.concat((neg_pad, tf.constant([[0, 0]])), axis=0))

                """ Output Target """
                # shape = (fh, fw, cls_num + 2)
                level_box_cls_target = tf.concat([level_box_cls_target, level_box_soft_weight[..., None],
                                                  level_box_regr_mask[..., None]], axis=-1)
                # shape = (fh, fw, 4 + 2)
                level_box_regr_target = tf.concat([level_box_regr_target, level_box_soft_weight[..., None],
                                                   level_box_regr_mask[..., None]], axis=-1)
                # level_box_pos_area = (dl + dr) * (dt + db)
                # (fh, fw)
                level_box_area = tf.pad(level_box_pos_area, neg_pad, constant_values=1e7)
                # level_box_area = tf.pad(level_pos_box_ap_weight ** 0.5, neg_pad, constant_values=0.)
                return level_box_cls_target, level_box_regr_target, level_box_area

            # cls_target : shape = (True_Label_count, fh, fw, cls_num + 2)
            # reg_target : shape = (True_Label_count, fh, fw, 4 + 2)
            # area : shape = (True_Label_count, fh, fw)
            level_cls_target, level_regr_target, level_area = tf.map_fn(
                build_map_function_target,
                elems=[pos_x1, pos_y1, pos_x2, pos_y2, gt_boxes, gt_labels, level_meta_select_weight],
                fn_output_signature=(tf.float32, tf.float32, tf.float32),
            )

            # min area : shape = (objects, fh, fw) --> (fh, fw)
            level_min_area_box_indices = tf.argmin(level_area, axis=0, output_type=tf.int32)
            # (fh, fw) --> (fh * fw)
            level_min_area_box_indices = tf.reshape(level_min_area_box_indices, (-1,))

            # (fw, ), (fh, )
            locs_x, locs_y = tf.range(0, fw), tf.range(0, fh)

            # (fh, fw) --> (fh * fw)
            locs_xx, locs_yy = tf.meshgrid(locs_x, locs_y)
            locs_xx = tf.reshape(locs_xx, (-1,))
            locs_yy = tf.reshape(locs_yy, (-1,))

            # (fh * fw, 3)
            level_indices = tf.stack((level_min_area_box_indices, locs_yy, locs_xx), axis=-1)

            """ Select """
            level_cls_target = tf.gather_nd(level_cls_target, level_indices)
            level_regr_target = tf.gather_nd(level_regr_target, level_indices)

            cls_target_ = tf.concat([cls_target_, level_cls_target], axis=0)
            regr_target_ = tf.concat([regr_target_, level_regr_target], axis=0)
        return [cls_target_, regr_target_]

    def false_function():
        fa = tf.reduce_prod(feature_maps_shape, axis=-1)
        fa_sum = tf.reduce_sum(fa)
        cls_target_ = tf.zeros((fa_sum, num_cls))
        regr_target_ = tf.zeros((fa_sum, 4))
        weight = tf.ones((fa_sum, 1))
        mask = tf.zeros((fa_sum, 1))
        cls_target_ = tf.concat([cls_target_, weight, mask], axis=-1)
        regr_target_ = tf.concat([regr_target_, weight, mask], axis=-1)
        return [cls_target_, regr_target_]

    cls_target, regr_target = tf.cond(
        tf.not_equal(tf.size(gt_boxes), 0),
        true_function,
        false_function
    )
    return [cls_target, regr_target]


def _build_map_function_module_target_iou(
        gt_boxes,
        feature_select_weight,
        feature_maps_shape,
        reg_pred,
        shrink_ratio=SR
):
    num_cls = NUM_CLS
    strides_ = (8, 16, 32, 64, 128)

    gt_labels = tf.cast(gt_boxes[:, 4], tf.int32)
    gt_boxes = gt_boxes[:, :4]
    gt_boxes, non_zeros = trim_zero_padding_boxes(gt_boxes)
    gt_labels = tf.boolean_mask(gt_labels, non_zeros)
    meta_select_weight = tf.boolean_mask(feature_select_weight, non_zeros)

    def _get_iou_score(target, pred):
        t_area = (target[..., 0] + target[..., 2]) * (target[..., 1] + target[..., 3])
        p_area = (pred[..., 0] + pred[..., 2]) * (pred[..., 1] + pred[..., 3])

        i_width = tf.minimum(target[..., 0], pred[..., 0]) + tf.minimum(target[..., 2], pred[..., 2])
        i_height = tf.minimum(target[..., 1], pred[..., 1]) + tf.minimum(target[..., 3], pred[..., 3])
        i_area = i_width * i_height

        u_area = t_area + p_area - i_area + 1e-7
        return i_area / u_area

    def true_function():
        cls_target_ = tf.zeros((0, num_cls + 2), dtype=tf.float32)
        regr_target_ = tf.zeros((0, 4 + 2), dtype=tf.float32)

        # P3 to P7
        for level_id in range(len(strides_)):
            # (objects, )
            level_meta_select_weight = meta_select_weight[:, level_id]

            # [8, 16, 32, 64, 128]: P3 to P7
            stride = strides_[level_id]

            # Feature map : height, width, area
            fh = feature_maps_shape[level_id][0]
            fw = feature_maps_shape[level_id][1]

            fa = tf.reduce_prod(feature_maps_shape, axis=-1)
            start_idx = tf.reduce_sum(fa[:level_id])
            end_idx = start_idx + fh * fw
            reg_pred_i = tf.reshape(reg_pred[start_idx:end_idx], (fh, fw, tf.shape(reg_pred)[-1]))

            pos_x1, pos_y1, pos_x2, pos_y2 = shrink_and_normalize_boxes(gt_boxes, fw, fh, stride, shrink_ratio)

            def build_map_function_target(args):
                pos_x1_ = args[0]
                pos_y1_ = args[1]
                pos_x2_ = args[2]
                pos_y2_ = args[3]
                gt_box = args[4]
                gt_label = args[5]
                level_box_meta_select_weight = args[6]

                """ Feature map from Regression Subnet """
                # (fh_p, fw_p, 4) --> (fh_p * fw_p, 4)
                locs_reg_pred_i = reg_pred_i[pos_y1_:pos_y2_, pos_x1_:pos_x2_, :]

                """ Create Negative sample """
                # padding top, bottom, left and right
                neg_top_bot = tf.stack((pos_y1_, fh - pos_y2_), axis=0)
                neg_lef_rit = tf.stack((pos_x1_, fw - pos_x2_), axis=0)
                neg_pad = tf.stack([neg_top_bot, neg_lef_rit], axis=0)

                """ Regression Target: create positive sample """
                pos_shift_xx = (tf.cast(tf.range(pos_x1_, pos_x2_), dtype=tf.float32) + 0.5) * stride
                pos_shift_yy = (tf.cast(tf.range(pos_y1_, pos_y2_), dtype=tf.float32) + 0.5) * stride
                pos_shift_xx, pos_shift_yy = tf.meshgrid(pos_shift_xx, pos_shift_yy)
                pos_shifts = tf.stack((pos_shift_xx, pos_shift_yy), axis=-1)
                dl = tf.maximum(pos_shifts[:, :, 0] - gt_box[0], 0)
                dt = tf.maximum(pos_shifts[:, :, 1] - gt_box[1], 0)
                dr = tf.maximum(gt_box[2] - pos_shifts[:, :, 0], 0)
                db = tf.maximum(gt_box[3] - pos_shifts[:, :, 1], 0)
                deltas = tf.stack((dl, dt, dr, db), axis=-1)
                level_box_regr_pos_target = deltas / 4.0 / stride  # (fh_p, fw_p, 4)
                level_pos_box_ap_weight = tf.minimum(dl, dr) * tf.minimum(dt, db) / tf.maximum(dl, dr) / tf.maximum(dt,
                                                                                                                    db)

                """ Computing IoU between target and predict """
                # gfl_cls_target = _get_iou_score2(pos_shifts, gt_box, locs_reg_pred_i, stride)
                gfl_cls_target = _get_iou_score(level_box_regr_pos_target, locs_reg_pred_i)
                level_pos_box_soft_weight = level_box_meta_select_weight * level_pos_box_ap_weight
                # level_pos_box_soft_weight = level_box_meta_select_weight * tf.ones_like(db)

                """ Classification Target: create positive sample """
                level_pos_box_cls_target = tf.zeros((pos_y2_ - pos_y1_, pos_x2_ - pos_x1_, num_cls), dtype=tf.float32)
                gfl_cls_target = tf.expand_dims(gfl_cls_target, axis=-1)
                level_pos_box_cls_target = tf.concat((level_pos_box_cls_target[..., :gt_label],
                                                      gfl_cls_target,
                                                      level_pos_box_cls_target[..., gt_label + 1:]), axis=-1)

                """ Padding Classification Target's negative sample """
                level_box_cls_target = tf.pad(level_pos_box_cls_target,
                                              tf.concat((neg_pad, tf.constant([[0, 0]])), axis=0))

                """ Padding Soft Anchor's negative sample """
                level_box_soft_weight = tf.pad(level_pos_box_soft_weight, neg_pad, constant_values=1)

                """ Creating Positive Sample locations and padding its negative sample """
                level_pos_box_regr_mask = tf.ones((pos_y2_ - pos_y1_, pos_x2_ - pos_x1_))
                level_box_regr_mask = tf.pad(level_pos_box_regr_mask, neg_pad)

                """ Padding Regression Target's negative sample """
                level_box_regr_target = tf.pad(level_box_regr_pos_target,
                                               tf.concat((neg_pad, tf.constant([[0, 0]])), axis=0))

                """ Output Target """
                # shape = (fh, fw, cls_num + 2)
                level_box_cls_target = tf.concat([level_box_cls_target, level_box_soft_weight[..., None],
                                                  level_box_regr_mask[..., None]], axis=-1)
                # shape = (fh, fw, 4 + 2)
                level_box_regr_target = tf.concat([level_box_regr_target, level_box_soft_weight[..., None],
                                                   level_box_regr_mask[..., None]], axis=-1)
                level_box_pos_area = (dl + dr) * (dt + db)
                # (fh, fw)
                level_box_area = tf.pad(level_box_pos_area, neg_pad, constant_values=1e7)
                return level_box_cls_target, level_box_regr_target, level_box_area

            # cls_target : shape = (True_Label_count, fh, fw, cls_num + 2)
            # reg_target : shape = (True_Label_count, fh, fw, 4 + 2)
            # area : shape = (True_Label_count, fh, fw)
            level_cls_target, level_regr_target, level_area = tf.map_fn(
                build_map_function_target,
                elems=[pos_x1, pos_y1, pos_x2, pos_y2, gt_boxes, gt_labels, level_meta_select_weight],
                fn_output_signature=(tf.float32, tf.float32, tf.float32),
            )

            # min area : shape = (objects, fh, fw) --> (fh, fw)
            level_min_area_box_indices = tf.argmin(level_area, axis=0, output_type=tf.int32)
            # (fh, fw) --> (fh * fw)
            level_min_area_box_indices = tf.reshape(level_min_area_box_indices, (-1,))

            # (fw, ), (fh, )
            locs_x, locs_y = tf.range(0, fw), tf.range(0, fh)

            # (fh, fw) --> (fh * fw)
            locs_xx, locs_yy = tf.meshgrid(locs_x, locs_y)
            locs_xx = tf.reshape(locs_xx, (-1,))
            locs_yy = tf.reshape(locs_yy, (-1,))

            # (fh * fw, 3)
            level_indices = tf.stack((level_min_area_box_indices, locs_yy, locs_xx), axis=-1)

            """ Select """
            level_cls_target = tf.gather_nd(level_cls_target, level_indices)
            level_regr_target = tf.gather_nd(level_regr_target, level_indices)

            cls_target_ = tf.concat([cls_target_, level_cls_target], axis=0)
            regr_target_ = tf.concat([regr_target_, level_regr_target], axis=0)
        return [cls_target_, regr_target_]

    def false_function():
        fa = tf.reduce_prod(feature_maps_shape, axis=-1)
        fa_sum = tf.reduce_sum(fa)
        cls_target_ = tf.zeros((fa_sum, num_cls))
        regr_target_ = tf.zeros((fa_sum, 4))
        weight = tf.ones((fa_sum, 1))
        mask = tf.zeros((fa_sum, 1))
        cls_target_ = tf.concat([cls_target_, weight, mask], axis=-1)
        regr_target_ = tf.concat([regr_target_, weight, mask], axis=-1)
        return [cls_target_, regr_target_]

    cls_target, regr_target = tf.cond(
        tf.not_equal(tf.size(gt_boxes), 0),
        true_function,
        false_function
    )
    return [cls_target, regr_target]


class Target(keras.layers.Layer):
    def __init__(self, num_cls=NUM_CLS, strides=STRIDES, shrink_ratio=SR, **kwargs):
        self.num_cls = num_cls,
        self.strides = strides,
        self.shrink_ratio = shrink_ratio
        super(Target, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        # (Batch, 5, 2)
        feature_map_shapes = inputs[0][0]

        # (Batch, Max_Bboxes_count, 5)
        batch_gt_boxes = inputs[1]

        # (Batch, Max_gt_boxes_count, 5)
        batch_feature_select_weight = inputs[2]

        batch_reg_pred = inputs[3]

        def _build_map_function_batch_module_target(args):
            """ For Batch axis. """
            gt_boxes, feature_select_weight, reg_pred = args[0], args[1], args[2]

            if config.USING_QFL:
                return _build_map_function_module_target_iou(
                    gt_boxes=gt_boxes,
                    feature_select_weight=feature_select_weight,
                    reg_pred=reg_pred,
                    shrink_ratio=self.shrink_ratio,
                    feature_maps_shape=feature_map_shapes
                )

            else:
                return _build_map_function_module_target(
                    gt_boxes=gt_boxes,
                    feature_select_weight=feature_select_weight,
                    shrink_ratio=self.shrink_ratio,
                    feature_maps_shape=feature_map_shapes
                )

        outputs = tf.map_fn(
            _build_map_function_batch_module_target,
            elems=[batch_gt_boxes, batch_feature_select_weight, batch_reg_pred],
            fn_output_signature=[tf.float32, tf.float32],
        )
        return outputs

    def get_config(self):
        c = super(Target, self).get_config()
        c.update({
            'num_cls': self.num_cls,
            'strides': self.strides,
            'shrink_ratio': self.shrink_ratio
        })
        return c


def freeze_model_bn(model_):
    for layer in model_.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False


def freeze_model_half(model_):
    for i, layer in enumerate(model_.layers):
        if i >= int(len(model_.layers) * 0.5):
            layer.trainable = True
            if isinstance(layer, keras_resnet.layers.BatchNormalization):
                layer.trainable = False
        else:
            layer.trainable = False


# 2020-10-14, Build BackBone
def _build_backbone(resnet=50, image_input=None, freeze_bn=False):
    setting = {
        'inputs': image_input,
        'include_top': False,
        'freeze_bn': freeze_bn
    }

    if resnet == 50:
        backbone = keras_resnet.models.ResNet50(**setting)

        if config.FREEZE_HALF_BACKBONE:
            freeze_model_half(backbone)

        if config.FREEZE_BACKBONE:
            backbone.trainable = False

    elif resnet == 101:
        backbone = keras_resnet.models.ResNet101(**setting)
        if config.FREEZE_HALF_BACKBONE:
            freeze_model_half(backbone)
        if config.FREEZE_BACKBONE:
            backbone.trainable = False

    elif resnet == 502:
        from tensorflow.keras.applications import ResNet50V2
        m = ResNet50V2(
            input_tensor=setting['inputs'],
            include_top=setting['include_top'],
            weights='imagenet' if config.PRETRAIN else None
        )
        if config.FREEZE_BN:
            freeze_model_bn(m)
        layer_names = ['conv3_block3_out', 'conv4_block5_out', 'conv5_block3_out']
        backbone = [m.get_layer(layer_name).output for layer_name in layer_names]

    elif resnet == 1012:
        from tensorflow.keras.applications import ResNet101V2
        m = ResNet101V2(
            input_tensor=setting['inputs'],
            include_top=setting['include_top'],
            weights='imagenet' if config.PRETRAIN else None
        )
        if config.FREEZE_BN:
            freeze_model_bn(m)
        layer_names = ['conv3_block3_out', 'conv4_block5_out', 'conv5_block3_out']
        backbone = [m.get_layer(layer_name).output for layer_name in layer_names]

    elif resnet == 503:
        from models.backbone import ResNet50
        backbone = ResNet50(inputs=image_input,
                            include_top=False,
                            gn=1
                            )

    else:
        raise ValueError('Wrong Backbone Name !')

    # if config.FREEZE_BACKBONE:
    #     backbone.trainable = False
    return backbone


def _build_head_subnets(input_features, width, depth, num_cls):
    cls_pred, reg_pred = [], []

    _setting = {
        'input_features': input_features,
        'width': width,
        'depth': depth,
        'num_cls': num_cls,
    }

    if config.HEAD == 'Mix':
        from models.head import MixSubnetworks
        cls_pred, reg_pred = MixSubnetworks(**_setting)

    elif config.HEAD == 'Std':
        from models.head import StdSubnetworks
        cls_pred, reg_pred = StdSubnetworks(**_setting)

    elif config.HEAD == 'Align':
        from models.head import AlignSubnetworks
        cls_pred, reg_pred = AlignSubnetworks(**_setting)

    return cls_pred, reg_pred


def _build_FSN(width):
    if config.FSN == 'V3':
        from models.neck import FSN
        return FSN(
            width=width, depth=3,
            fpn_level=len(STRIDES), ws=config.FSN_WS
        )


MODEL_CONFIG = {
    "FeatureSelectInput": FeatureSelectInput,
    "FeatureSelectTarget": FeatureSelectTarget,
    "FSNLoss": FSNLoss,
    "FeatureSelectWeight_V2": FeatureSelectWeight_V2,
    "FocalLoss": FocalLoss,
    "IoULoss": IoULoss,
    "Target": Target,
    "_build_backbone": _build_backbone,
    "create_pyramid_features_v2": create_pyramid_features_v2,
    "_build_head_subnets": _build_head_subnets,
    "_build_FSN": _build_FSN
}


# 2020-10-14, Feature Select Model.
def SAPD(
        soft=False,
        num_cls=20,
        max_gt_boxes=100,
        width=256,
        depth=4,
        resnet=50,
        freeze_bn=False
):
    """
        soft : True for using soft weight, False for Top-1 weight
        num_cls : number of classes
        max_gt_boxes: maximum of total bboxes
    """
    # Image Input: 影像, 限定Channel.
    image_input = keras.layers.Input((None, None, 3), name='image')
    # Gt Boxes Input: 真實框, [x1, y1, x2, y2, class]
    gt_boxes_input = keras.layers.Input((max_gt_boxes, 5), name='bboxes')
    # True Label Gt Boxes Count Input: 真實框數量
    true_label_gt_boxes_count_input = keras.layers.Input((1,), dtype=tf.int32, name='bboxes_count')
    # Feature Maps Shape Input: 各FPN輸出特徵圖大小
    feature_maps_shape_input = keras.layers.Input((5, 2), dtype=tf.int32, name='fmaps_shape')

    """ Creating Backbone """
    backbone = _build_backbone(resnet=resnet, image_input=image_input, freeze_bn=freeze_bn)
    C3, C4, C5 = backbone.outputs[1:] if config.BACKBONE_TYPE == 'ResNetV1' else backbone

    """ Creating Feature Pyramid Network """
    features = create_pyramid_features_v2(C3, C4, C5, n=0)

    """ Creating Subnetworks """
    model_pred = _build_head_subnets(
        input_features=features,
        width=width,
        depth=depth,
        num_cls=num_cls
    )

    """ Feature Selection Network """
    feature_select_model = _build_FSN(width=width)

    """ Creating FSN Input """
    feature_select_input, batch_gt_boxes_id = FeatureSelectInput(
        pool_size=config.FSN_POOL_SIZE
    )([gt_boxes_input, *features])
    feature_select_pred = feature_select_model(feature_select_input)

    """ Target for FSN """
    feature_select_target = FeatureSelectTarget()(
        [model_pred[0], model_pred[-1], feature_maps_shape_input, gt_boxes_input]
    )

    """ Loss for FSN """
    feature_select_loss = FSNLoss(
        factor=config.FSN_FACTOR,
        name='fsn_loss'
    )([feature_select_target, feature_select_pred])

    """ Soft Anchor-point weights """
    weight = FeatureSelectWeight_V2(max_gt_boxes_count=max_gt_boxes, soft=soft)([
        feature_select_pred if soft else feature_select_target,
        batch_gt_boxes_id,
        true_label_gt_boxes_count_input
    ])

    """ Target for Subnetworks """
    cls_target, reg_target = Target(num_cls=num_cls)(
        [feature_maps_shape_input, gt_boxes_input, weight, model_pred[-1]]
    )

    """ Loss for Subnetworks """
    cls_loss = FocalLoss(
        name='cls_loss'
    )([cls_target, model_pred[0]])
    reg_loss = IoULoss(
        mode=config.IOU_LOSS,
        factor=config.IOU_FACTOR,
        name='loc_loss'
    )([reg_target, model_pred[-1]])

    training_model = keras.models.Model(
        inputs=[image_input, gt_boxes_input, true_label_gt_boxes_count_input, feature_maps_shape_input],
        outputs=[cls_loss, reg_loss, feature_select_loss],
        name='training_model'
    )

    if config.EVALUATION:
        """ Inference """
        locs, strides = Locations2()(features)
        boxes = RegressionBoxes2(name='boxes')([locs, strides, model_pred[-1]])
        boxes = ClipBoxes2(name='clip_boxes')([image_input, boxes])
        detections = FilterDetections2(
            nms=1 if config.NMS == 1 else 0,
            s_nms=1 if config.NMS == 2 else 0,
            nms_threshold=config.NMS_TH,
            name='filtered_detections',
            score_threshold=config.SCORE_TH,
            max_detections=config.DETECTIONS,
        )([boxes, model_pred[0]])

        prediction_model = keras.models.Model(
            inputs=[image_input],
            outputs=detections,
            name='inference_model'
        )
        """ Training and Inference """
        return training_model, prediction_model

    else:
        """ Only Training """
        prediction_model = None
        return training_model, prediction_model
