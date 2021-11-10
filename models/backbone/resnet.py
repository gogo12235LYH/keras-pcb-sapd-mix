import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.regularizers
from tensorflow_addons.layers import GroupNormalization

parameters = {
    "kernel_initializer": "he_normal"
}


def bottleneck_2d(
        filters,
        stage=0,
        block=0,
        kernel_size=3,
        numerical_name=False,
        stride=None,
        bn=0,
        gn=0,
        groups=16,
):
    if stride is None:
        if block != 0 or stage == 0:
            stride = 1
        else:
            stride = 2

    if keras.backend.image_data_format() == "channels_last":
        axis = 3
    else:
        axis = 1

    if block > 0 and numerical_name:
        block_char = "b{}".format(block)
    else:
        block_char = chr(ord('a') + block)

    stage_char = str(stage + 2)

    def f(x):
        y = keras.layers.Conv2D(filters, (1, 1), strides=stride, use_bias=False,
                                name="res{}{}_branch2a".format(stage_char, block_char), **parameters)(x)

        if bn:
            y = keras.layers.BatchNormalization(axis=axis, epsilon=1e-5,
                                                name="bn{}{}_branch2a".format(stage_char, block_char))(y)
        elif gn:
            y = GroupNormalization(epsilon=1e-5, groups=groups,
                                   name="gn{}{}_branch2a".format(stage_char, block_char))(y)

        y = keras.layers.Activation("relu", name="res{}{}_branch2a_relu".format(stage_char, block_char))(y)

        y = keras.layers.ZeroPadding2D(padding=1, name="padding{}{}_branch2b".format(stage_char, block_char))(y)

        y = keras.layers.Conv2D(filters, kernel_size, use_bias=False,
                                name="res{}{}_branch2b".format(stage_char, block_char), **parameters)(y)

        if bn:
            y = keras.layers.BatchNormalization(axis=axis, epsilon=1e-5,
                                                name="bn{}{}_branch2b".format(stage_char, block_char))(y)
        elif gn:
            y = GroupNormalization(epsilon=1e-5, groups=groups,
                                   name="gn{}{}_branch2b".format(stage_char, block_char))(y)

        y = keras.layers.Activation("relu", name="res{}{}_branch2b_relu".format(stage_char, block_char))(y)

        y = keras.layers.Conv2D(filters * 4, (1, 1), use_bias=False,
                                name="res{}{}_branch2c".format(stage_char, block_char), **parameters)(y)

        if bn:
            y = keras.layers.BatchNormalization(axis=axis, epsilon=1e-5,
                                                name="bn{}{}_branch2c".format(stage_char, block_char))(y)
        elif gn:
            y = GroupNormalization(epsilon=1e-5, groups=groups,
                                   name="gn{}{}_branch2c".format(stage_char, block_char))(y)

        if block == 0:
            shortcut = keras.layers.Conv2D(filters * 4, (1, 1), strides=stride, use_bias=False,
                                           name="res{}{}_branch1".format(stage_char, block_char), **parameters)(x)

            if bn:
                shortcut = keras.layers.BatchNormalization(axis=axis, epsilon=1e-5,
                                                           name="bn{}{}_branch1".format(stage_char, block_char))(
                    shortcut)

            elif gn:
                shortcut = GroupNormalization(epsilon=1e-5, groups=groups,
                                              name="gn{}{}_branch1".format(stage_char, block_char))(shortcut)

        else:
            shortcut = x

        y = keras.layers.Add(name="res{}{}".format(stage_char, block_char))([y, shortcut])

        y = keras.layers.Activation("relu", name="res{}{}_relu".format(stage_char, block_char))(y)

        return y

    return f


class ResNet2D(tensorflow.keras.Model):
    """
    Constructs a `keras.models.Model` object using the given block count.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

    :param blocks: the network’s residual architecture

    :param block: a residual block (e.g. an instance of `keras_resnet.blocks.basic_2d`)

    :param include_top: if true, includes classification layers

    :param classes: number of classes to classify (include_top must be true)

    :param freeze_bn: if true, freezes BatchNormalization layers (ie. no updates are done in these layers)

    :param numerical_names: list of bool, same size as blocks, used to indicate whether names of layers should include numbers or letters

    :return model: ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)

    """

    def __init__(
            self,
            inputs,
            blocks,
            block,
            include_top=True,
            classes=1000,
            bn=0,
            gn=0,
            groups=16,
            numerical_names=None,
            *args,
            **kwargs
    ):
        if keras.backend.image_data_format() == "channels_last":
            axis = 3
        else:
            axis = 1

        if numerical_names is None:
            numerical_names = [True] * len(blocks)

        x = keras.layers.Conv2D(64, (7, 7), strides=(2, 2), use_bias=False, name="conv1", padding="same")(inputs)

        if bn:
            x = keras.layers.BatchNormalization(axis=axis, epsilon=1e-5, name="bn_conv1")(x)
        elif gn:
            x = GroupNormalization(epsilon=1e-5, groups=groups, name="gn_conv1")(x)

        x = keras.layers.Activation("relu", name="conv1_relu")(x)
        x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same", name="pool1")(x)

        features = 64

        outputs = []

        for stage_id, iterations in enumerate(blocks):
            for block_id in range(iterations):
                x = block(
                    features,
                    stage_id,
                    block_id,
                    numerical_name=(block_id > 0 and numerical_names[stage_id]),
                    bn=bn,
                    gn=gn,
                    groups=groups,
                )(x)

            features *= 2

            outputs.append(x)

        if include_top:
            assert classes > 0

            x = keras.layers.GlobalAveragePooling2D(name="pool5")(x)
            x = keras.layers.Dense(classes, activation="softmax", name="fc1000")(x)

            super(ResNet2D, self).__init__(inputs=inputs, outputs=x, *args, **kwargs)
        else:
            # Else output each stages features
            super(ResNet2D, self).__init__(inputs=inputs, outputs=outputs, *args, **kwargs)


class ResNet2D50(ResNet2D):
    """
    Constructs a `keras.models.Model` according to the ResNet50 specifications.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

    :param blocks: the network’s residual architecture

    :param include_top: if true, includes classification layers

    :param classes: number of classes to classify (include_top must be true)

    :param freeze_bn: if true, freezes BatchNormalization layers (ie. no updates are done in these layers)

    :return model: ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)


    """

    def __init__(self, inputs, blocks=None, include_top=True, classes=1000, groups=32, *args, **kwargs):
        if blocks is None:
            blocks = [3, 4, 6, 3]

        numerical_names = [False, False, False, False]

        super(ResNet2D50, self).__init__(
            inputs,
            blocks,
            numerical_names=numerical_names,
            block=bottleneck_2d,
            include_top=include_top,
            classes=classes,
            groups=groups,
            *args,
            **kwargs
        )


class ResNet2D101(ResNet2D):
    """
    Constructs a `keras.models.Model` according to the ResNet101 specifications.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

    :param blocks: the network’s residual architecture

    :param include_top: if true, includes classification layers

    :param classes: number of classes to classify (include_top must be true)

    :return model: ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)

    """

    def __init__(self, inputs, blocks=None, include_top=True, classes=1000, groups=32, freeze_bn=False, *args,
                 **kwargs):
        if blocks is None:
            blocks = [3, 4, 23, 3]

        numerical_names = [False, True, True, False]

        super(ResNet2D101, self).__init__(
            inputs,
            blocks,
            numerical_names=numerical_names,
            block=bottleneck_2d,
            include_top=include_top,
            classes=classes,
            groups=groups,
            freeze_bn=freeze_bn,
            *args,
            **kwargs
        )


if __name__ == '__main__':
    model = ResNet2D101(
        inputs=tf.keras.layers.Input((512, 512, 3)),
        include_top=False,
        gn=1,
    )

    model.summary()
    # tf.keras.utils.plot_model(model, show_shapes=True)
