import tensorflow_datasets as tfds
import tensorflow as tf


def preprocess_data(sample):
    """Applies preprocessing step to a single sample

    ref: https://keras.io/examples/vision/retinanet/#preprocessing-data

    Arguments:
      sample: A dict representing a single training sample.

    Returns:
      image: Resized and padded image with random horizontal flipping applied.
      bbox: Bounding boxes with the shape `(num_objects, 4)` where each box is
        of the format `[x, y, width, height]`.
      class_id: An tensor representing the class id of the objects, having
        shape `(num_objects,)`.
    """

    # image_input
    image = tf.cast(sample["image"], dtype=tf.float32)
    bbox = sample["objects"]["bbox"]
    class_id = tf.cast(sample["objects"]["label"], dtype=tf.float32)

    image_shape = tf.cast(tf.shape(image)[:2], dtype=tf.float32)

    # gt_boxes_input
    bbox = tf.stack(
        [
            bbox[:, 0] * image_shape[1],
            bbox[:, 1] * image_shape[0],
            bbox[:, 2] * image_shape[1],
            bbox[:, 3] * image_shape[0],
            class_id
        ],
        axis=-1,
    )

    # true_label_count
    bbox_num = tf.shape(bbox[0])

    # fmaps_shape
    fmaps_shape = tf.constant(
        [[80, 80], [40, 40], [20, 20], [10, 10], [5, 5]],
        dtype=tf.int32
    )

    return image, bbox, bbox_num, fmaps_shape


if __name__ == '__main__':
    batch_size = 16

    print("[INFO] Batch size : ", batch_size)

    autotune = tf.data.AUTOTUNE

    train, test = tfds.load(name="dpcb_db", split=["train", "test"], data_dir="D:/datasets/")

    train = train.map(preprocess_data, num_parallel_calls=autotune)

    train = train.padded_batch(
        batch_size=batch_size, padding_values=(0.0, 1e-8, 0, 0), drop_remainder=True
    )

    train = train.prefetch(autotune)

    tfds.benchmark(train, batch_size=batch_size)

    tfds.benchmark(train, batch_size=batch_size)

    #     [INFO] Batch size :  16
    # ************ Summary ************
    #
    # Examples/sec (First included) 429.93 ex/sec (total: 992 ex, 2.31 sec)
    # Examples/sec (First only) 138.36 ex/sec (total: 16 ex, 0.12 sec)
    # Examples/sec (First excluded) 445.31 ex/sec (total: 976 ex, 2.19 sec)
    #
    # ************ Summary ************
    #
    # Examples/sec (First included) 454.25 ex/sec (total: 992 ex, 2.18 sec)
    # Examples/sec (First only) 187.91 ex/sec (total: 16 ex, 0.09 sec)
    # Examples/sec (First excluded) 465.06 ex/sec (total: 976 ex, 2.10 sec)
