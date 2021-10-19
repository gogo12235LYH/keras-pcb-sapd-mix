import cv2
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

_image_size = [512, 640, 768, 896, 1024, 1280, 1408]


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
        fmap_shape = int(_img_size / _strides[i])
        shapes.append([fmap_shape, fmap_shape])

    return shapes


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
    if tf.random.uniform(()) > prob:
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
    rotate_k = np.random.choice([1, 2, 3])

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

    if tf.random.uniform(()) > prob:
        image = tf.image.rot90(image, k=rotate_k)

        bboxes = tf.map_fn(
            _rotate_bbox,
            elems=bboxes,
            fn_output_signature=tf.float32
        )

    return image, bboxes


def preprocess_data(
        phi: int = 0,
        mode: str = "ResNetV1",
        fmap_shapes: any = None,
        max_bboxes: int = 100,
        padding_value: float = 0.
):
    """Applies preprocessing step to a single sample

    ref: https://keras.io/examples/vision/retinanet/#preprocessing-data

    """

    def _resize_image(image, target_size=512):
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
        image = tf.pad(image, pad, constant_values=padding_value)

        return image, scale, offset_h, offset_w

    def _padding_bboxes(bboxes, classes, scale, offset_h, offset_w):

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
        max_bbox_pad = tf.stack([tf.stack([tf.constant(0), max_bboxes - bboxes_count], axis=0),
                                 tf.constant([0, 0]),
                                 ], axis=0)
        bboxes = tf.pad(bboxes, max_bbox_pad, constant_values=0.)
        return bboxes, bboxes_count

    def _preprocess_data(sample):
        #
        image, image_shape, bboxes, classes = compute_inputs(sample)

        # TODO: data augmentation
        image, bboxes = tf_rotate(image, image_shape, bboxes, prob=0.5)
        image, bboxes = random_flip_horizontal(image, image_shape, bboxes, prob=0.5)

        #
        image, scale, offset_h, offset_w = _resize_image(image=image, target_size=_image_size[phi])
        image = _normalization_image(image, mode)
        bboxes, bboxes_count = _padding_bboxes(bboxes, classes, scale, offset_h, offset_w)
        fmaps_shape = tf.constant(fmap_shapes, dtype=tf.int32)
        return image, bboxes, bboxes_count[None], fmaps_shape

    return _preprocess_data


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


def inputs_targets(image, bboxes, bboxes_count, fmaps_shape):
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


def create_pipeline(phi=0, mode="ResNetV1", db="DPCB", batch_size=1):
    autotune = tf.data.AUTOTUNE

    if db == "DPCB":
        (train, test) = tfds.load(name="dpcb_db", split=["train", "test"], data_dir="D:/datasets/")
    else:
        train = None
        test = None

    feature_maps_shapes = _fmap_shapes(phi)

    train = train.map(preprocess_data(
        phi=phi,
        mode=mode,
        fmap_shapes=feature_maps_shapes
    ), num_parallel_calls=autotune)

    train = train.shuffle(8 * batch_size)
    train = train.padded_batch(batch_size=batch_size, padding_values=(0.0, 0.0, 0, 0), drop_remainder=True)

    train = train.map(inputs_targets, num_parallel_calls=autotune)

    train = train.prefetch(autotune).repeat()

    return train, test


def create_pipeline_test(phi=0, mode="ResNetV1", db="DPCB", batch_size=1):
    autotune = tf.data.AUTOTUNE

    if db == "DPCB":
        (train, test) = tfds.load(name="dpcb_db", split=["train", "test"], data_dir="D:/huai/datasets/")
    else:
        train = None
        test = None

    feature_maps_shapes = _fmap_shapes(phi)

    train = train.map(preprocess_data(
        phi=phi,
        mode=mode,
        fmap_shapes=feature_maps_shapes
    ), num_parallel_calls=autotune)

    train = train.shuffle(8 * batch_size)
    train = train.padded_batch(batch_size=batch_size, padding_values=(0.0, 0.0, 0, 0), drop_remainder=True)
    train = train.map(inputs_targets, num_parallel_calls=autotune)
    train = train.prefetch(autotune)
    return train, test


if __name__ == '__main__':
    bs = 4

    train_t, test_t = create_pipeline_test(
        phi=1,
        batch_size=bs
    )

    tfds.benchmark(train_t, batch_size=bs)

    tfds.benchmark(train_t, batch_size=bs)
