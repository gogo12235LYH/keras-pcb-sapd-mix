import config
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow.keras as keras
import tensorflow.keras.backend as k
from tensorflow.python.keras.utils.data_utils import get_file
from generators.voc import PascalVocGenerator
from preprocess.color_aug import VisualEffect
from preprocess.misc_aug import MiscEffect
from eval.voc import Evaluate
from models import SAPD
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow_addons.optimizers import SGDW, AdamW
from models.losses import model_loss
from generators import load_test
from callbacks import create_callbacks


def _init():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def download_ResNet_imagenet_w(depth=50):
    """
        Downloads ImageNet weights and returns path to weights file.
    """
    resnet_filename = 'ResNet-{}-model.keras.h5'
    resnet_resource = 'https://github.com/fizyr/keras-models/releases/download/v0.0.1/{}'.format(resnet_filename)

    filename = resnet_filename.format(depth)
    resource = resnet_resource.format(depth)

    if depth == 50:
        checksum = '3e9f4e4f77bbe2c9bec13b53ee1c2319'
    elif depth == 101:
        checksum = '05dc86924389e5b401a9ea0348a3213c'
    elif depth == 152:
        checksum = '6ee11ef2b135592f8031058820bb9e71'
    elif depth == 502:
        return None

    else:
        raise ValueError('Unknown depth')

    return get_file(
        filename,
        resource,
        cache_subdir='models',
        md5_hash=checksum
    )


def load_weights(input_model, model_name):
    if config.MODE == 1:

        # Stage 1
        if config.PRETRAIN == 1:
            if config.BACKBONE_TYPE == 'ResNetV1':
                weight = download_ResNet_imagenet_w(depth=config.BACKBONE)
                print(f' Imagenet Pretrain ... ', end='')
                input_model.load_weights(weight, by_name=True)
                print("OK.")
            else:
                print(f" Imagenet ... OK.")

        elif config.PRETRAIN == 2:
            print(f" Pretrain weight from {config.PRETRAIN_WEIGHT} ... ", end='')
            input_model.load_weights(config.PRETRAIN_WEIGHT, by_name=True)
            print("OK.")

    elif config.MODE == 2:

        # Stage 2
        weight = f"{model_name}.h5"
        print(f" {weight} ... ", end='')
        input_model.load_weights(weight, by_name=True)
        print("OK.")

    else:
        # Stage Top-1 or Top-k
        if config.PRETRAIN == 1:
            if config.BACKBONE_TYPE == 'ResNetV1':
                weight = download_ResNet_imagenet_w(depth=config.BACKBONE)
                print(f" Imagenet Pretrain ... ", end='')
                input_model.load_weights(weight, by_name=True)
                print("OK.")
            else:
                print(f" Imagenet ... OK.")


def create_generators(
        batch_size=2,
        phi=0,
        path=r'../VOCdevkit/VOC2012+2007',
        misc_aug=True,
        visual_aug=True
):
    """
    Create generators for training and validation.
    Args
        args: parseargs object containing configuration for generators.
        preprocess_image: Function that preprocesses an image for the network.
    """
    common_args = {
        'batch_size': batch_size,
        'phi': phi,
    }

    misc_effect = MiscEffect() if misc_aug else None
    visual_effect = VisualEffect() if visual_aug else None

    # train_generator_ = PascalVocGenerator(
    #     path,
    #     'trainval' if config.MixUp_AUG == 0 else 'trainval_mixup',
    #     skip_difficult=True,
    #     misc_effect=misc_effect,
    #     visual_effect=visual_effect,
    #     **common_args
    # )

    autotune = tf.data.AUTOTUNE
    (train, test) = tfds.load(name="dpcb_db", split=["train", "test"], data_dir="D:/datasets/")
    train = train.map(load_test.preprocess_data, num_parallel_calls=autotune)
    train = train.shuffle(8 * batch_size)
    train = train.padded_batch(
        batch_size=batch_size, padding_values=(0.0, 0.0, 0, 0), drop_remainder=True
    )
    train = train.map(load_test.inputs_targets, num_parallel_calls=autotune)
    train_generator_ = train.prefetch(autotune).repeat()

    test_generator_ = PascalVocGenerator(
        path,
        'test',
        skip_difficult=True,
        shuffle_groups=False,
        **common_args
    )

    return train_generator_, test_generator_


def create_optimizer(opt_name, base_lr, m, decay):
    if opt_name == 'SGD':
        return keras.optimizers.SGD(
            learning_rate=base_lr,
            momentum=m,
            decay=decay,
            nesterov=config.USE_NESTEROV
        )

    if opt_name == 'Adam':
        return keras.optimizers.Adam(
            learning_rate=base_lr
        )

    if opt_name == 'SGDW':
        return SGDW(
            learning_rate=base_lr,
            weight_decay=decay,
            momentum=m,
            nesterov=config.USE_NESTEROV
        )

    if opt_name == 'AdamW':
        opt = AdamW(
            learning_rate=base_lr,
            weight_decay=decay
        )
        return opt

    raise ValueError("[INFO] Got WRONG Optimizer name. PLZ CHECK again !!")


def create_model_name(info_, epochs_, phi_, backbone_, depth_, batch_):
    return f"{info_}-E{epochs_}BS{batch_}B{phi_}R{backbone_}D{depth_}"


def model_compile(info, model_name, optimizer):
    print(f"{info} Creating Model... ")
    model_, pred_model_ = SAPD(
        soft=True if config.MODE in [2, 4] else False,
        num_cls=config.NUM_CLS,
        depth=config.SUBNET_DEPTH,
        resnet=config.BACKBONE,
        freeze_bn=config.FREEZE_BN
    )

    print(f"{info} Loading Weight... ", end='')
    load_weights(input_model=model_, model_name=model_name)

    print(f"{info} Model Compiling... ")
    model_.compile(
        optimizer=optimizer,
        loss=model_loss(
            cls='cls_loss',
            reg='reg_loss',
            fsn='feature_select_loss'
        )
    )
    return model_, pred_model_


def main():
    print("[INFO] Initializing... ")
    _init()

    stage = f"[INFO] Stage {config.MODE} :"

    """ Optimizer Setup """
    print(f"{stage} Creating Optimizer... ")
    Optimizer = create_optimizer(
        opt_name=config.OPTIMIZER,
        base_lr=config.BASE_LR,
        m=config.MOMENTUM,
        decay=config.DECAY
    )

    print(f"{stage} Creating Generators... ")
    train_generator, test_generator = create_generators(
        batch_size=config.BATCH_SIZE,
        phi=config.PHI,
        misc_aug=config.MISC_AUG,
        visual_aug=config.VISUAL_AUG,
        path=config.DATABASE_PATH,
    )

    """ Multi GPU Accelerating"""
    if config.MULTI_GPU:
        # mirrored_strategy = tf.distribute.MirroredStrategy()
        mirrored_strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
        with mirrored_strategy.scope():
            model, pred_model = model_compile(stage, config.NAME, Optimizer)
    else:
        model, pred_model = model_compile(stage, config.NAME, Optimizer)

    print(f"{stage} Model Name : {config.NAME}")

    print(f"{stage} Creating Callbacks... ", end='')
    callbacks = create_callbacks(
        config=config,
        pred_mod=pred_model,
        test_gen=test_generator,
    )

    """ Training, the batch size of generator is global batch size. """
    """ Ex: If global batch size and GPUs are 32 and 4, it is 8 (32/4) images per GPU during training. """
    print(f"{stage} Start Training... ")
    model.fit(
        train_generator,
        epochs=config.EPOCHs_STAGE_ONE if config.MODE == 1 else config.EPOCHs,
        initial_epoch=config.EPOCHs_STAGE_ONE - 1 if config.MODE == 2 else 0,
        callbacks=callbacks,
        steps_per_epoch=config.STEPs_PER_EPOCH,
    )

    """ Save model's weights """
    save_model_name = config.NAME + "-soft" if config.MODE == 2 else config.NAME
    print(f"{stage} Saving Model Weights : {save_model_name}.h5 ... ")
    model.save_weights(f'{save_model_name}.h5')


if __name__ == '__main__':
    main()
