import tensorflow as tf


def resize_images(images, size, method='nearest', anti=False):
    methods = {
        'bilinear': tf.image.ResizeMethod.BILINEAR,
        'lanczos3': tf.image.ResizeMethod.LANCZOS3,
        'lanczos5': tf.image.ResizeMethod.LANCZOS5,
        'bicubic': tf.image.ResizeMethod.BICUBIC,
        'nearest': tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        'area': tf.image.ResizeMethod.AREA,
        'mitchellcubic': tf.image.ResizeMethod.MITCHELLCUBIC
    }
    return tf.image.resize(images, size, methods[method], antialias=anti)


@tf.function(jit_compile=True)
def xyxy2cxcywh(xyxy):
    """
    :param xyxy: (x1, y1, x2, y2)
    :return: x1, y1, x2, y2 to Xc, Yc, w, h
    """
    return tf.concat((0.5 * (xyxy[:, :2] + xyxy[:, 2:4]), xyxy[:, 2:4] - xyxy[:, :2]), axis=-1)


@tf.function(jit_compile=True)
def cxcywh2xyxy(cxcywh):
    """
    :param cxcywh: (Xc, Yc, w, h)
    :return: (Xc, Yc, w, h) to (x1, y1, x2, y2)
    """
    return tf.concat((cxcywh[:, :2] - 0.5 * cxcywh[:, 2:4], cxcywh[:, :2] + 0.5 * cxcywh[:, 2:4]), axis=-1)


@tf.function(jit_compile=True)
def normalize_boxes(boxes, width, height, stride):
    """
    :param boxes:
    :param width:
    :param height:
    :param stride:
    :return:
    """
    x1 = boxes[:, 0:1] / stride / width
    y1 = boxes[:, 1:2] / stride / height
    x2 = boxes[:, 2:3] / stride / width
    y2 = boxes[:, 3:4] / stride / height
    return tf.concat([x1, y1, x2, y2], axis=-1)


@tf.function(jit_compile=True, experimental_relax_shapes=True)
def shrink_and_normalize_boxes(boxes, width, height, stride, shrink_ratio=0.2, kd=False):
    boxes = xyxy2cxcywh(boxes)
    boxes = tf.concat((boxes[:, :2], boxes[:, 2:4] * shrink_ratio), axis=-1)
    boxes = cxcywh2xyxy(boxes)

    width = tf.cast(width, tf.float32)
    height = tf.cast(height, tf.float32)

    if kd:
        x1 = tf.floor(boxes[:, 0:1] / stride)
        y1 = tf.floor(boxes[:, 1:2] / stride)
        x2 = tf.math.ceil(boxes[:, 2:3] / stride)
        y2 = tf.math.ceil(boxes[:, 3:4] / stride)

    else:
        x1 = tf.floor(boxes[:, 0] / stride)
        y1 = tf.floor(boxes[:, 1] / stride)
        x2 = tf.math.ceil(boxes[:, 2] / stride)
        y2 = tf.math.ceil(boxes[:, 3] / stride)

    x2 = tf.cast(tf.clip_by_value(x2, 1, width), tf.int32)
    y2 = tf.cast(tf.clip_by_value(y2, 1, height), tf.int32)
    x1 = tf.cast(tf.clip_by_value(x1, 0, tf.cast(x2, tf.float32) - 1), tf.int32)
    y1 = tf.cast(tf.clip_by_value(y1, 0, tf.cast(y2, tf.float32) - 1), tf.int32)
    return x1, y1, x2, y2


@tf.function
def create_reg_positive_sample(bboxes, x1, y1, x2, y2, stride):
    shift_xx = (tf.cast(tf.range(x1, x2), dtype=tf.float32) + 0.5) * stride
    shift_yy = (tf.cast(tf.range(y1, y2), dtype=tf.float32) + 0.5) * stride
    shift_xx, shift_yy = tf.meshgrid(shift_xx, shift_yy)
    shifts = tf.stack((shift_xx, shift_yy), axis=-1)

    lef = tf.maximum(shifts[..., 0] - bboxes[0], 0)
    top = tf.maximum(shifts[..., 1] - bboxes[1], 0)
    rit = tf.maximum(bboxes[2] - shifts[..., 0], 0)
    bot = tf.maximum(bboxes[3] - shifts[..., 1], 0)

    reg_target = tf.stack((lef, top, rit, bot), axis=-1) / 4.0 / stride
    anchor_pots = tf.minimum(lef, rit) * tf.minimum(top, bot) / tf.maximum(lef, rit) / tf.maximum(top, bot)
    area = (lef + rit) * (top + bot)

    return reg_target, anchor_pots, area


def trim_zero_padding_boxes(boxes):
    """
    :param boxes: (Max_Boxes, 4)
    :return:

    輸入的boxes數量有100，

    """
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=-1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros)
    return boxes, non_zeros


if __name__ == '__main__':
    pass
    # print(feature_maps_shape_gen(1, 5))
