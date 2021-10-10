import tensorflow as tf
import tensorflow_datasets as tfds


@tf.function
def _preprocess_image(image, image_size: int, mode: str = "ResNetV1", padding_value: float = 128.):
    image_height, image_width = tf.shape(image)[1], tf.shape(image)[2]

    if image_height > image_width:
        scale = tf.cast((image_size / image_height), dtype=tf.float32)
        resized_height = image_size
        resized_width = tf.cast((tf.cast(image_width, dtype=tf.float32) * scale), dtype=tf.int32)
    else:
        scale = tf.cast((image_size / image_width), dtype=tf.float32)
        resized_width = image_size
        resized_height = tf.cast((tf.cast(image_height, dtype=tf.float32) * scale), dtype=tf.int32)

    image = tf.image.resize(image, (resized_height, resized_width))
    offset_h = (image_size - resized_height) // 2
    offset_w = (image_size - resized_width) // 2

    # (h, w, c)
    pad = tf.stack(
        [
            tf.stack([offset_h, offset_h], axis=0),
            tf.stack([offset_w, offset_w], axis=0),
            tf.constant([0, 0]),
        ],
        axis=0
    )
    image = tf.pad(image, pad, constant_values=padding_value)

    # image normalization
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

    return image, scale, offset_h, offset_h


def _fmap_shapes(phi: int = 0, level: int = 5):
    _image_size = int(phi * 128) + 512
    _strides = [int(2 ** (x + 3)) for x in range(level)]

    shapes = []

    for i in range(level):
        fmap_shape = int(_image_size / _strides[i])
        shapes.append([fmap_shape, fmap_shape])

    return shapes


def random_flip_horizontal(image, boxes):
    """Flips image and boxes horizontally with 50% chance

    Arguments:
      image: A 3-D tensor of shape `(height, width, channels)` representing an
        image.
      boxes: A tensor with shape `(num_boxes, 4)` representing bounding boxes,
        having normalized coordinates.

    Returns:
      Randomly flipped image and boxes
    """
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        boxes = tf.stack(
            [1 - boxes[:, 2],
             boxes[:, 1],
             1 - boxes[:, 0],
             boxes[:, 3]]
            , axis=-1
        )
    return image, boxes


def preprocess_data(phi=0, mode="ResNetV1", max_bboxes=100, padding_value=0.):
    """Applies preprocessing step to a single sample

    ref: https://keras.io/examples/vision/retinanet/#preprocessing-data

    """

    image_size = [512, 640, 768, 896, 1024, 1280, 1408]
    fmap_shapes = _fmap_shapes(phi)

    @tf.function
    def _preprocess_data(sample):
        # image_input
        image = tf.cast(sample["image"], dtype=tf.float32)
        image_shape = tf.cast(tf.shape(image)[:2], dtype=tf.float32)
        bbox = tf.cast(sample["objects"]["bbox"], dtype=tf.float32)
        class_id = tf.cast(sample["objects"]["label"], dtype=tf.float32)

        # TODO: data augmentation
        image, bbox = random_flip_horizontal(image, bbox)

        image, scale, offset_h, offset_w = _preprocess_image(
            image=image,
            image_size=image_size[phi],
            mode=mode,
            padding_value=padding_value,
        )

        # gt_boxes_input
        bbox = tf.stack(
            [
                bbox[:, 0] * image_shape[1] * scale + tf.cast(offset_w, dtype=tf.float32),
                bbox[:, 1] * image_shape[0] * scale + tf.cast(offset_h, dtype=tf.float32),
                bbox[:, 2] * image_shape[1] * scale + tf.cast(offset_w, dtype=tf.float32),
                bbox[:, 3] * image_shape[0] * scale + tf.cast(offset_h, dtype=tf.float32),
                class_id
            ],
            axis=-1,
        )

        # true_label_count
        bbox_num = tf.shape(bbox)[0]

        max_bbox_pad = tf.stack(
            [
                tf.stack([tf.constant(0), max_bboxes - bbox_num], axis=0),
                tf.constant([0, 0]),
            ],
            axis=0
        )
        bbox = tf.pad(bbox, max_bbox_pad, constant_values=0.)

        # fmaps_shape
        fmaps_shape = tf.constant(fmap_shapes, dtype=tf.int32)

        return image, bbox, bbox_num[None], fmaps_shape

    return _preprocess_data


def inputs_targets(image, bbox, bbox_num, fmaps_shape):
    inputs = {
        "image_input": image,
        "gt_boxes_input": bbox,
        "true_label_count": bbox_num,
        "fmaps_shape": fmaps_shape,
    }

    targets = [
        tf.zeros([tf.shape(image)[0], ], dtype=tf.float32),
        tf.zeros([tf.shape(image)[0], ], dtype=tf.float32),
        tf.zeros([tf.shape(image)[0], ], dtype=tf.float32),
    ]

    return inputs, targets
