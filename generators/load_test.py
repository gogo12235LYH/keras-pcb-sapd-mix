import tensorflow_datasets as tfds
import tensorflow as tf
from generators.data_pipline import preprocess_data, inputs_targets


# from utils.util_graph import resize_images_pipeline


if __name__ == '__main__':
    batch_size = 8

    print("[INFO] Batch size : ", batch_size)

    autotune = tf.data.AUTOTUNE

    (train, test) = tfds.load(name="dpcb_db", split=["train", "test"], data_dir="D:/datasets/")

    train = train.map(preprocess_data(phi=1, max_bboxes=15), num_parallel_calls=autotune)

    train = train.shuffle(8 * batch_size)

    train = train.padded_batch(
        batch_size=batch_size, padding_values=(0.0, 0.0, 0, 0), drop_remainder=True
    )

    train = train.map(inputs_targets, num_parallel_calls=autotune)

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
