import config
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from utils.util_graph import shrink_and_normalize_boxes, create_reg_positive_sample

_image_size = [512, 640, 768, 896, 1024, 1280, 1408]
_STRIDES = [8, 16, 32, 64, 128]


def _normalization_image(image, mode):
    if mode == 'ResNetV1':
        # Caffe
        image = image[..., ::-1]  # RGB -> BGR
        image -= [103.939, 116.779, 123.68]

    elif mode == 'ResNetV2':
        image /= 127.5
        image -= 1.

    elif mode == 'EffNet':
        image = image

    elif mode in ['DenseNet', 'SEResNet']:
        # Torch
        image /= 255.
        image -= [0.485, 0.456, 0.406]
        image /= [0.229, 0.224, 0.225]

    return image


def _fmap_shapes(phi: int = 0, level: int = 5):
    _img_size = int(phi * 128) + 512
    _strides = [int(2 ** (x + 3)) for x in range(level)]

    shapes = []

    for i in range(level):
        fmap_shape = _img_size // _strides[i]
        shapes.append([fmap_shape, fmap_shape])

    return shapes


@tf.function
def random_flip_horizontal(
        image, image_shape, bboxes, prob=0.5
):
    """Flips image and boxes horizontally

    Arguments:
      image: A 3-D tensor of shape `(height, width, channels)` representing an
        image.
      image_shape:
      bboxes: A tensor with shape `(num_boxes, 4)` representing bounding boxes,
        having normalized coordinates.
      prob: Chance.

    Returns:
      Randomly flipped image and boxes
    """
    if tf.random.uniform((), dtype=tf.float16) > prob:
        image = tf.image.flip_left_right(image)
        bboxes = tf.stack(
            [image_shape[1] - bboxes[:, 2],
             bboxes[:, 1],
             image_shape[1] - bboxes[:, 0],
             bboxes[:, 3]]
            , axis=-1
        )
    return image, bboxes


@tf.function
def tf_rotate(
        image, image_shape, bboxes, prob=0.5
):
    offset = image_shape / 2.
    rotate_k = tf.random.uniform((), minval=1, maxval=4, dtype=tf.int32)

    def _r_method(x, y, angle):
        tf_cos = tf.math.cos(angle)
        tf_sin = tf.math.sin(angle)

        tf_abs_cos = tf.abs(tf_cos)
        tf_abs_sin = tf.abs(tf_sin)

        offset_h, offset_w = offset[0], offset[1]

        new_offset_w = offset_w * (tf_abs_cos - tf_cos) + offset_h * (tf_abs_sin - tf_sin)
        new_offset_h = offset_w * (tf_abs_sin + tf_sin) + offset_h * (tf_abs_cos - tf_cos)

        x_r = x * tf_cos + y * tf_sin + new_offset_w
        y_r = x * tf_sin * -1 + y * tf_cos + new_offset_h

        return x_r, y_r

    def _rotate_bbox(bbox):
        # degree: pi/2, pi, 3*pi/2
        angle = tf.cast(rotate_k, dtype=tf.float32) * (np.pi / 2.)

        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

        x1_n, y1_n = _r_method(x1, y1, angle)
        x2_n, y2_n = _r_method(x2, y2, angle)

        bbox = tf.stack([
            tf.minimum(x1_n, x2_n), tf.minimum(y1_n, y2_n),
            tf.maximum(x1_n, x2_n), tf.maximum(y1_n, y2_n)
        ])
        return bbox

    if tf.random.uniform((), dtype=tf.float16) > prob:
        image = tf.image.rot90(image, k=rotate_k)

        bboxes = tf.map_fn(
            _rotate_bbox,
            elems=bboxes,
            fn_output_signature=tf.float32
        )
    return image, bboxes


@tf.function
def multi_scale(
        image, image_shape, bboxes, prob=0.5
):
    if tf.random.uniform((), dtype=tf.float16) > prob:
        # start, end, step = 0.25, 1.3, 0.05
        # scale = np.random.choice(np.arange(start, end, step))
        scale = tf.random.uniform((), minval=0.6, maxval=2.0)

        new_image_shape = tf.cast(image_shape * scale, dtype=tf.int32)
        image = tf.image.resize(image, new_image_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        image_shape *= scale
        bboxes *= scale
    return image, image_shape, bboxes


@tf.function
def random_crop(
        image, image_shape, bboxes, prob=0.5
):
    if tf.random.uniform((), dtype=tf.float16) > prob:
        min_x1y1 = tf.math.reduce_min(bboxes, axis=0)[:2]
        max_x2y2 = tf.math.reduce_max(bboxes, axis=0)[2:]

        random_x1 = tf.random.uniform((), minval=0, maxval=tf.maximum(min_x1y1[0] / 2., 1.), dtype=tf.float32)
        random_y1 = tf.random.uniform((), minval=0, maxval=tf.maximum(min_x1y1[1] / 2., 1.), dtype=tf.float32)

        random_x2 = tf.random.uniform(
            (),
            minval=max_x2y2[0] - 1.0,
            maxval=tf.math.maximum(
                tf.math.minimum(image_shape[1], max_x2y2[0] + (image_shape[1] - max_x2y2[0]) / 2.),
                max_x2y2[0]
            ),
            dtype=tf.float32
        )
        random_y2 = tf.random.uniform(
            (),
            minval=max_x2y2[1] - 1.,
            maxval=tf.math.maximum(
                tf.math.minimum(image_shape[0], max_x2y2[1] + (image_shape[0] - max_x2y2[1]) / 2.),
                max_x2y2[1]
            ),
            dtype=tf.float32
        )

        image = tf.image.crop_to_bounding_box(
            image,
            offset_height=tf.cast(random_y1, dtype=tf.int32),
            offset_width=tf.cast(random_x1, dtype=tf.int32),
            target_height=tf.cast(random_y2 - random_y1, dtype=tf.int32),
            target_width=tf.cast(random_x2 - random_x1, dtype=tf.int32)
        )

        bboxes = tf.stack(
            [
                bboxes[:, 0] - random_x1,
                bboxes[:, 1] - random_y1,
                bboxes[:, 2] - random_x1,
                bboxes[:, 3] - random_y1,
            ],
            axis=-1
        )

    return image, bboxes


def random_image_saturation(image, prob=.5):
    if tf.random.uniform(()) > prob:
        image = tf.image.random_saturation(image)

    return image


def bboxes_clip(bboxes, image_shape):
    bboxes = tf.stack(
        [
            tf.clip_by_value(bboxes[:, 0], 0, image_shape[1] - 2),  # x1
            tf.clip_by_value(bboxes[:, 1], 0, image_shape[0] - 2),  # y1
            tf.clip_by_value(bboxes[:, 2], 1, image_shape[1] - 1),  # x2
            tf.clip_by_value(bboxes[:, 3], 1, image_shape[0] - 1),  # y2
        ],
        axis=-1
    )
    return bboxes


def compute_inputs(sample):
    image = tf.cast(sample["image"], dtype=tf.float32)
    image_shape = tf.cast(tf.shape(image)[:2], dtype=tf.float32)
    bboxes = tf.cast(sample["objects"]["bbox"], dtype=tf.float32)
    classes = tf.cast(sample["objects"]["label"], dtype=tf.float32)

    bboxes = tf.stack(
        [
            bboxes[:, 0] * image_shape[1],
            bboxes[:, 1] * image_shape[0],
            bboxes[:, 2] * image_shape[1],
            bboxes[:, 3] * image_shape[0],
        ],
        axis=-1
    )
    return image, image_shape, bboxes, classes


def preprocess_data(
        phi: int = 0,
        mode: str = "ResNetV1",
        fmap_shapes: any = None,
        max_bboxes: int = 100,
        padding_value: float = 128.,
        debug: bool = False,
):
    """Applies preprocessing step to a single sample

    ref: https://keras.io/examples/vision/retinanet/#preprocessing-data

    """

    def _resize_image(image, target_size=512):
        image_height, image_width = tf.shape(image)[0], tf.shape(image)[1]

        # if image_height > image_width:
        #     scale = tf.cast((target_size / image_height), dtype=tf.float32)
        #     resized_height = target_size
        #     resized_width = tf.cast((tf.cast(image_width, dtype=tf.float32) * scale), dtype=tf.int32)
        # else:
        #     scale = tf.cast((target_size / image_width), dtype=tf.float32)
        #     resized_height = tf.cast((tf.cast(image_height, dtype=tf.float32) * scale), dtype=tf.int32)
        #     resized_width = target_size

        if image_height > target_size or image_width > target_size:
            if image_height > image_width:
                scale = tf.cast((target_size / image_height), dtype=tf.float32)
                resized_height = target_size
                resized_width = tf.cast((tf.cast(image_width, dtype=tf.float32) * scale), dtype=tf.int32)
            else:
                scale = tf.cast((target_size / image_width), dtype=tf.float32)
                resized_height = tf.cast((tf.cast(image_height, dtype=tf.float32) * scale), dtype=tf.int32)
                resized_width = target_size

            image = tf.image.resize(
                image,
                (resized_height, resized_width),
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        else:
            resized_height = image_height
            resized_width = image_width
            scale = 1.0

        offset_h = (target_size - resized_height) // 2
        offset_w = (target_size - resized_width) // 2

        # (h, w, c)
        pad = tf.stack([tf.stack([offset_h, offset_h], axis=0),
                        tf.stack([offset_w, offset_w], axis=0),
                        tf.constant([0, 0]),
                        ], axis=0)
        image = tf.pad(image, pad, constant_values=padding_value)

        return image, scale, offset_h, offset_w

    def _padding_bboxes(bboxes, classes, scale, offset_h, offset_w, padding=True):

        const = float(_image_size[phi]) if debug else 1.

        # gt_boxes_input
        bboxes = tf.stack(
            [
                bboxes[:, 0] * scale + tf.cast(offset_w, dtype=tf.float32) / const,
                bboxes[:, 1] * scale + tf.cast(offset_h, dtype=tf.float32) / const,
                bboxes[:, 2] * scale + tf.cast(offset_w, dtype=tf.float32) / const,
                bboxes[:, 3] * scale + tf.cast(offset_h, dtype=tf.float32) / const,
                classes
            ],
            axis=-1,
        )
        bboxes = tf.clip_by_value(bboxes, 0., 1. if debug else float(_image_size[phi]))

        if padding:
            # true_label_count
            bboxes_count = tf.shape(bboxes)[0]
            max_bbox_pad = tf.stack([tf.stack([tf.constant(0), max_bboxes - bboxes_count], axis=0),
                                     tf.constant([0, 0]),
                                     ], axis=0)
            bboxes = tf.pad(bboxes, max_bbox_pad, constant_values=0.)

        else:
            bboxes_count = tf.shape(bboxes)[0]

        return bboxes, bboxes_count

    def _preprocess_data(sample):
        #
        image, image_shape, bboxes, classes = compute_inputs(sample)

        # Data augmentation
        image, image_shape, bboxes = multi_scale(image, image_shape, bboxes, prob=0.5)
        image, bboxes = tf_rotate(image, image_shape, bboxes, prob=0.5)
        image, bboxes = random_flip_horizontal(image, image_shape, bboxes, prob=0.5)
        image, bboxes = random_crop(image, image_shape, bboxes, prob=0.5)

        # Clip Bboxes
        bboxes = bboxes_clip(bboxes, image_shape)

        #
        image, scale, offset_h, offset_w = _resize_image(image=image, target_size=_image_size[phi])
        image = _normalization_image(image, mode) if not debug else image

        bboxes, bboxes_count = _padding_bboxes(bboxes, classes, scale, offset_h, offset_w, padding=True)

        fmaps_shape = tf.constant(fmap_shapes, dtype=tf.int32)
        return image, bboxes, bboxes_count[None], fmaps_shape

    return _preprocess_data


def preprocess_data_v2(
        phi: int = 0,
        mode: str = "ResNetV1",
        fmap_shapes: any = None,
        padding_value: float = 128.,
        debug: bool = False,
):
    """Applies preprocessing step to a single sample

    ref: https://keras.io/examples/vision/retinanet/#preprocessing-data

    """

    def _resize_image(image, target_size=512):
        image_height, image_width = tf.shape(image)[0], tf.shape(image)[1]

        # if image_height > image_width:
        #     scale = tf.cast((target_size / image_height), dtype=tf.float32)
        #     resized_height = target_size
        #     resized_width = tf.cast((tf.cast(image_width, dtype=tf.float32) * scale), dtype=tf.int32)
        # else:
        #     scale = tf.cast((target_size / image_width), dtype=tf.float32)
        #     resized_height = tf.cast((tf.cast(image_height, dtype=tf.float32) * scale), dtype=tf.int32)
        #     resized_width = target_size

        if image_height > target_size or image_width > target_size:
            if image_height > image_width:
                scale = tf.cast((target_size / image_height), dtype=tf.float32)
                resized_height = target_size
                resized_width = tf.cast((tf.cast(image_width, dtype=tf.float32) * scale), dtype=tf.int32)
            else:
                scale = tf.cast((target_size / image_width), dtype=tf.float32)
                resized_height = tf.cast((tf.cast(image_height, dtype=tf.float32) * scale), dtype=tf.int32)
                resized_width = target_size

            image = tf.image.resize(
                image,
                (resized_height, resized_width),
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        else:
            resized_height = image_height
            resized_width = image_width
            scale = 1.0

        offset_h = (target_size - resized_height) // 2
        offset_w = (target_size - resized_width) // 2

        # (h, w, c)
        pad = tf.stack([tf.stack([offset_h, offset_h], axis=0),
                        tf.stack([offset_w, offset_w], axis=0),
                        tf.constant([0, 0]),
                        ], axis=0)
        image = tf.pad(image, pad, constant_values=padding_value)

        return image, scale, offset_h, offset_w

    def _padding_bboxes(bboxes, classes, scale, offset_h, offset_w):

        const = float(_image_size[phi]) if debug else 1.

        # gt_boxes_input
        bboxes = tf.stack(
            [
                bboxes[:, 0] * scale + tf.cast(offset_w, dtype=tf.float32) / const,
                bboxes[:, 1] * scale + tf.cast(offset_h, dtype=tf.float32) / const,
                bboxes[:, 2] * scale + tf.cast(offset_w, dtype=tf.float32) / const,
                bboxes[:, 3] * scale + tf.cast(offset_h, dtype=tf.float32) / const,
                classes
            ],
            axis=-1,
        )
        bboxes = tf.clip_by_value(bboxes, 0., 1. if debug else float(_image_size[phi]))
        return bboxes

    def _preprocess_data(sample):
        #
        image, image_shape, bboxes, classes = compute_inputs(sample)

        # Data augmentation
        image, image_shape, bboxes = multi_scale(image, image_shape, bboxes, prob=0.5)
        image, bboxes = tf_rotate(image, image_shape, bboxes, prob=0.5)
        image, bboxes = random_flip_horizontal(image, image_shape, bboxes, prob=0.5)
        image, bboxes = random_crop(image, image_shape, bboxes, prob=0.5)

        # Clip Bboxes
        bboxes = bboxes_clip(bboxes, image_shape)

        #
        image, scale, offset_h, offset_w = _resize_image(image=image, target_size=_image_size[phi])
        image = _normalization_image(image, mode) if not debug else image

        bboxes = _padding_bboxes(bboxes, classes, scale, offset_h, offset_w)

        fmaps_shape = tf.constant(fmap_shapes, dtype=tf.int32)
        return image, bboxes[:, :4], bboxes[:, -1], fmaps_shape

    return _preprocess_data


@tf.function
def _compute_targets(image, bboxes, classes, fmap_shapes):
    num_cls = config.NUM_CLS

    cls_target_ = tf.zeros((0, num_cls + 2), dtype=tf.float32)
    reg_target_ = tf.zeros((0, 4 + 2), dtype=tf.float32)
    ind_target_ = tf.zeros((0, 1), dtype=tf.int32)

    classes = tf.cast(classes, tf.int32)

    for level in range(len(_STRIDES)):
        stride = _STRIDES[level]

        fh = fmap_shapes[level][0]
        fw = fmap_shapes[level][1]

        pos_x1, pos_y1, pos_x2, pos_y2 = shrink_and_normalize_boxes(bboxes, fh, fw, stride, config.SHRINK_RATIO)

        def build_map_function_target(args):
            pos_x1_ = args[0]
            pos_y1_ = args[1]
            pos_x2_ = args[2]
            pos_y2_ = args[3]
            box = args[4]
            cls = args[5]

            """ Create Negative sample """
            neg_top_bot = tf.stack((pos_y1_, fh - pos_y2_), axis=0)
            neg_lef_rit = tf.stack((pos_x1_, fw - pos_x2_), axis=0)
            neg_pad = tf.stack([neg_top_bot, neg_lef_rit], axis=0)

            """ Regression Target: create positive sample """
            _loc_target, _ap_weight, _area = create_reg_positive_sample(
                box, pos_x1_, pos_y1_, pos_x2_, pos_y2_, stride
            )

            """ Classification Target: create positive sample """
            _cls_target = tf.zeros((pos_y2_ - pos_y1_, pos_x2_ - pos_x1_, num_cls), dtype=tf.float32)
            _cls_onehot = tf.ones((pos_y2_ - pos_y1_, pos_x2_ - pos_x1_, 1), dtype=tf.float32)
            _cls_target = tf.concat((_cls_target[..., :cls], _cls_onehot, _cls_target[..., cls + 1:]), axis=-1)

            """ Padding Classification Target's negative sample """
            _cls_target = tf.pad(_cls_target, tf.concat((neg_pad, tf.constant([[0, 0]])), axis=0))

            """ Padding Soft Anchor's negative sample """
            _ap_weight = tf.pad(_ap_weight, neg_pad, constant_values=1)

            """ Creating Positive Sample locations and padding it's negative sample """
            _pos_mask = tf.ones((pos_y2_ - pos_y1_, pos_x2_ - pos_x1_))
            _pos_mask = tf.pad(_pos_mask, neg_pad)

            """ Padding Regression Target's negative sample """
            _loc_target = tf.pad(_loc_target, tf.concat((neg_pad, tf.constant([[0, 0]])), axis=0))

            """ Output Target """
            # shape = (fh, fw, cls_num + 2)
            _cls_target = tf.concat([_cls_target, _ap_weight[..., None], _pos_mask[..., None]], axis=-1)
            # shape = (fh, fw, 4 + 2)
            _loc_target = tf.concat([_loc_target, _ap_weight[..., None], _pos_mask[..., None]], axis=-1)
            # (fh, fw)
            _area = tf.pad(_area, neg_pad, constant_values=1e7)

            return _cls_target, _loc_target, _area

        # cls_target : shape = (anchor-points, fh, fw, cls_num + 2)
        # reg_target : shape = (anchor-points, fh, fw, 4 + 2)
        # area : shape = (anchor-points, fh, fw)
        level_cls_target, level_reg_target, level_area = tf.map_fn(
            build_map_function_target,
            elems=[pos_x1, pos_y1, pos_x2, pos_y2, bboxes, classes],
            fn_output_signature=(tf.float32, tf.float32, tf.float32),
        )

        # min area : shape = (targets, fh, fw) --> (fh, fw)
        level_min_area_indices = tf.argmin(level_area, axis=0, output_type=tf.int32)
        # (fh, fw) --> (fh * fw)
        level_min_area_indices = tf.reshape(level_min_area_indices, (-1,))

        # (fw, ), (fh, )
        locs_x, locs_y = tf.range(0, fw), tf.range(0, fh)

        # (fh, fw) --> (fh * fw)
        locs_xx, locs_yy = tf.meshgrid(locs_x, locs_y)
        locs_xx = tf.reshape(locs_xx, (-1,))
        locs_yy = tf.reshape(locs_yy, (-1,))

        # (fh * fw, 3)
        level_indices = tf.stack((level_min_area_indices, locs_yy, locs_xx), axis=-1)

        """ Select """
        level_cls_target = tf.gather_nd(level_cls_target, level_indices)
        level_reg_target = tf.gather_nd(level_reg_target, level_indices)

        cls_target_ = tf.concat([cls_target_, level_cls_target], axis=0)
        reg_target_ = tf.concat([reg_target_, level_reg_target], axis=0)
        ind_target_ = tf.concat([ind_target_, tf.expand_dims(level_min_area_indices, -1)], axis=0)

    ind_target_ = tf.where(
        tf.equal(cls_target_[..., -1], 1.), ind_target_[..., 0], -1
    )[..., None]

    # Shape: (anchor-points, cls_num + 2) and (anchor-points, 4 + 2)
    # return image, cls_target_, reg_target_
    return image, cls_target_, reg_target_, ind_target_, tf.shape(bboxes)[0][..., None]


def inputs_targets(image, bboxes, bboxes_count, fmaps_shape):
    inputs = {
        "image": image,
        "bboxes": bboxes,
        "bboxes_count": bboxes_count,
        "fmaps_shape": fmaps_shape,
    }
    return inputs


# def inputs_targets_v2(image, cls_target, reg_target):
def inputs_targets_v2(image, cls_target, reg_target, ind_target, bboxes_cnt):
    # image, cls_target, reg_target
    inputs = {
        "image": image,
        "cls_target": cls_target,
        "loc_target": reg_target,
        "ind_target": ind_target,
        "bboxes_cnt": bboxes_cnt
    }
    return inputs


def create_pipeline(phi=0, mode="ResNetV1", db="DPCB", batch_size=1):
    autotune = tf.data.AUTOTUNE

    if db == "DPCB":
        (train, test) = tfds.load(name="dpcb_db", split=["train", "test"], data_dir="C:/works/datasets/")
    else:
        train = None
        test = None

    feature_maps_shapes = _fmap_shapes(phi)

    train = train.map(preprocess_data(
        phi=phi,
        mode=mode,
        fmap_shapes=feature_maps_shapes
    ), num_parallel_calls=autotune)

    train = train.shuffle(1000).repeat()
    train = train.padded_batch(batch_size=batch_size, padding_values=(0.0, 0.0, 0, 0), drop_remainder=True)
    train = train.map(inputs_targets, num_parallel_calls=autotune)
    train = train.prefetch(autotune)
    return train, test


def create_pipeline_v2(phi=0, mode="ResNetV1", db="DPCB", batch_size=1, debug=False):
    autotune = tf.data.AUTOTUNE

    if db == "DPCB":
        (train, test) = tfds.load(name="dpcb_db", split=["train", "test"], data_dir="C:/works/datasets/")
    else:
        train = None
        test = None

    feature_maps_shapes = _fmap_shapes(phi)

    train = train.map(preprocess_data_v2(
        phi=phi,
        mode=mode,
        fmap_shapes=feature_maps_shapes
    ), num_parallel_calls=autotune)

    train = train.shuffle(train.__len__())
    train = train.map(_compute_targets, num_parallel_calls=autotune)
    # train = train.padded_batch(batch_size=batch_size, padding_values=(0.0, 0.0, 0.0,), drop_remainder=True)
    train = train.padded_batch(batch_size=batch_size, padding_values=(0.0, 0.0, 0.0, 0, 0), drop_remainder=True)
    train = train.map(inputs_targets_v2, num_parallel_calls=autotune)

    if debug:
        train = train.prefetch(autotune)
    else:
        train = train.repeat().prefetch(autotune)
    return train, test


def create_pipeline_test(phi=0, mode="ResNetV1", db="DPCB", batch_size=1):
    autotune = tf.data.AUTOTUNE

    if db == "DPCB":
        (train, test) = tfds.load(name="dpcb_db", split=["train", "test"], data_dir="C:/works/datasets/")
    else:
        train = None
        test = None

    feature_maps_shapes = _fmap_shapes(phi)

    train = train.map(preprocess_data(
        phi=phi,
        mode=mode,
        fmap_shapes=feature_maps_shapes,
        max_bboxes=16,
        debug=True
    ), num_parallel_calls=autotune)

    train = train.shuffle(train.__len__())
    train = train.padded_batch(batch_size=batch_size, padding_values=(0.0, 0.0, 0, 0), drop_remainder=True)
    train = train.map(inputs_targets, num_parallel_calls=autotune)
    train = train.prefetch(autotune)
    return train, test


class PipeLine:
    def __init__(
            self,
            phi: int = 0,
            batch_size: int = 2,
            database_name: str = "DPCB",
            database_path: str = "D:/datasets/",
            mode: str = "ResNetV1",
            misc_aug: bool = False,
            color_aug: bool = False,
            max_bboxes: int = 100,
            padding_value: float = 0.
    ):
        self.phi = phi
        self.batch_size = batch_size
        self.database_name = database_name
        self.database_path = database_path
        self.mode = mode
        self.misc_aug = misc_aug
        self.color_aug = color_aug
        self.max_bboxes = max_bboxes
        self.padding_value = padding_value

        self.db_dict = {
            "DPCB": "dpcb_db"
        }

    def create(self, test_mode: bool = False):

        autotune = tf.data.AUTOTUNE

        train, test = self._load_data()

        # inputs
        train = train.map(self._compute_inputs, num_parallel_calls=autotune)

        # augmentation
        train = train.map(self._augmentation, num_parallel_calls=autotune)

        # targets
        train = train.map(self._compute_targets, num_parallel_calls=autotune)

        # setting batch size and padding
        train = train.shuffle(8 * self.batch_size)
        train = train.padded_batch(batch_size=self.batch_size, padding_values=(0.0, 0.0, 0, 0), drop_remainder=True)

        # combine inputs and targets
        train = train.map(self._inputs_targets, num_parallel_calls=autotune)
        train = train.prefetch(autotune).repeat() if not test_mode else train.prefetch(autotune)
        return train, test

    def _load_data(self):
        (train, test) = tfds.load(name=self.db_dict[self.database_name], split=["train", "test"],
                                  data_dir=self.database_path)
        return train, test

    @staticmethod
    def _compute_inputs(sample):
        image = tf.cast(sample["image"], dtype=tf.float32)
        image_shape = tf.cast(tf.shape(image)[:2], dtype=tf.float32)
        bboxes = tf.cast(sample["objects"]["bbox"], dtype=tf.float32)
        classes = tf.cast(sample["objects"]["label"], dtype=tf.float32)

        bboxes = tf.stack(
            [
                bboxes[:, 0] * image_shape[1],
                bboxes[:, 1] * image_shape[0],
                bboxes[:, 2] * image_shape[1],
                bboxes[:, 3] * image_shape[0],
            ],
            axis=-1
        )
        return image, image_shape, bboxes, classes

    def _augmentation(self, image, image_shape, bboxes, classes):
        if not self.misc_aug and not self.color_aug:
            return image, bboxes, classes

        else:
            if self.misc_aug:
                image, bboxes = tf_rotate(image, image_shape, bboxes, prob=0.5)
                image, bboxes = random_flip_horizontal(image, image_shape, bboxes, prob=0.5)

            if self.color_aug:
                pass

            return image, bboxes, classes

    def _compute_targets(self, image, bboxes, classes):
        image, scale, offset_h, offset_w = self._resize_image_offset(image=image, target_size=_image_size[self.phi])

        image = _normalization_image(image, mode=self.mode)

        bboxes, bboxes_count = self._padding_bboxes(bboxes, classes, scale, offset_h, offset_w)

        fmaps_shape = tf.constant(_fmap_shapes(self.phi), dtype=tf.int32)

        return image, bboxes, bboxes_count[None], fmaps_shape

    def _resize_image_offset(self, image, target_size=512):
        image_height, image_width = tf.shape(image)[0], tf.shape(image)[1]

        if image_height > image_width:
            scale = tf.cast((target_size / image_height), dtype=tf.float32)
            resized_height = target_size
            resized_width = tf.cast((tf.cast(image_width, dtype=tf.float32) * scale), dtype=tf.int32)
        else:
            scale = tf.cast((target_size / image_width), dtype=tf.float32)
            resized_height = tf.cast((tf.cast(image_height, dtype=tf.float32) * scale), dtype=tf.int32)
            resized_width = target_size

        image = tf.image.resize(image, (resized_height, resized_width), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        offset_h = (target_size - resized_height) // 2
        offset_w = (target_size - resized_width) // 2

        # (h, w, c)
        pad = tf.stack([tf.stack([offset_h, offset_h], axis=0),
                        tf.stack([offset_w, offset_w], axis=0),
                        tf.constant([0, 0]),
                        ], axis=0)
        image = tf.pad(image, pad, constant_values=self.padding_value)

        return image, scale, offset_h, offset_w

    def _padding_bboxes(self, bboxes, classes, scale, offset_h, offset_w):

        # gt_boxes_input
        bboxes = tf.stack(
            [
                bboxes[:, 0] * scale + tf.cast(offset_w, dtype=tf.float32),
                bboxes[:, 1] * scale + tf.cast(offset_h, dtype=tf.float32),
                bboxes[:, 2] * scale + tf.cast(offset_w, dtype=tf.float32),
                bboxes[:, 3] * scale + tf.cast(offset_h, dtype=tf.float32),
                classes
            ],
            axis=-1,
        )

        # true_label_count
        bboxes_count = tf.shape(bboxes)[0]
        max_bbox_pad = tf.stack([tf.stack([tf.constant(0), self.max_bboxes - bboxes_count], axis=0),
                                 tf.constant([0, 0]),
                                 ], axis=0)
        bboxes = tf.pad(bboxes, max_bbox_pad, constant_values=0.)
        return bboxes, bboxes_count

    @staticmethod
    def _inputs_targets(image, bboxes, bboxes_count, fmaps_shape):
        inputs = {
            "image": image,
            "bboxes": bboxes,
            "bboxes_count": bboxes_count,
            "fmaps_shape": fmaps_shape,
        }

        targets = [
            tf.zeros([tf.shape(image)[0], ], dtype=tf.float32),
            tf.zeros([tf.shape(image)[0], ], dtype=tf.float32),
            tf.zeros([tf.shape(image)[0], ], dtype=tf.float32),
        ]

        return inputs, targets


if __name__ == '__main__':
    bs = 4

    train_t, test_t = create_pipeline_v2(
        phi=1,
        batch_size=bs,
        debug=True
    )

    """
    
        # batch size with 8 on 640 by 640:
    
        # ************ Summary ************
        # Examples/sec (First included) 181.46 ex/sec (total: 1000 ex, 5.51 sec)
        # Examples/sec (First only) 1.96 ex/sec (total: 8 ex, 4.08 sec)
        # Examples/sec (First excluded) 693.80 ex/sec (total: 992 ex, 1.43 sec)
        # ************ Summary ************
        # Examples/sec (First included) 188.66 ex/sec (total: 1000 ex, 5.30 sec)
        # Examples/sec (First only) 2.07 ex/sec (total: 8 ex, 3.86 sec)
        # Examples/sec (First excluded) 687.46 ex/sec (total: 992 ex, 1.44 sec)
    
    """

    iterations = 1
    for step, inputs_batch in enumerate(train_t):
        if (step + 1) > iterations:
            break

        print(f"[INFO] {step + 1} / {iterations}")

        _cls = inputs_batch['cls_target'].numpy()
        _loc = inputs_batch['loc_target'].numpy()
        _ind = inputs_batch['ind_target'].numpy()
        _int = inputs_batch['bboxes_cnt'].numpy()

    # import matplotlib.pyplot as plt
    #
    # iterations = 10
    # plt.figure(figsize=(10, 8))
    #
    # for step, inputs_batch in enumerate(train_t):
    #     if (step + 1) > iterations:
    #         break
    #
    #     print(f"[INFO] {step + 1} / {iterations}")
    #
    #     images = inputs_batch['image']
    #     bboxes = inputs_batch['bboxes']
    #
    #     bboxes = tf.stack(
    #         [
    #             bboxes[..., 1],
    #             bboxes[..., 0],
    #             bboxes[..., 3],
    #             bboxes[..., 2],
    #         ],
    #         axis=-1
    #     )
    #
    #     colors = np.array([[255.0, 0.0, 0.0]])
    #     images = tf.image.draw_bounding_boxes(
    #         images,
    #         bboxes,
    #         colors=colors
    #     )
    #
    #     for i in range(bs):
    #         plt.subplot(2, 2, i + 1)
    #         plt.imshow(images[i].numpy().astype("uint8"))
    #         # print(bboxes[i])
    #
    #     plt.pause(0.001)

    # tfds.benchmark(train_t, batch_size=bs)
    # tfds.benchmark(train_t, batch_size=bs)

    # image : (Batch, None, None, 3)
    # bboxes : (Batch, None, 5)
    # bboxes_count : (Batch, 1)
    # fmaps_shape : (Batch, 5, 2)
