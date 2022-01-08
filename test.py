import tensorflow as tf
from tensorflow.python.keras.utils.data_utils import get_file
from generators.voc import PascalVocGenerator
from preprocess.color_aug import VisualEffect
from preprocess.misc_aug import MiscEffect
import tensorflow.keras as keras
from models.sapd import SAPD
from tensorflow_addons.optimizers import SGDW, AdamW
from models.losses import model_loss
import config


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


def create_optimizer(opt_name, base_lr, m, decay):
    if opt_name == 'SGD':
        return keras.optimizers.SGD(
            lr=base_lr,
            momentum=m,
            decay=decay,
            nesterov=config.USE_NESTEROV
        )

    if opt_name == 'Adam':
        return keras.optimizers.Adam(
            lr=base_lr
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
    model_, pred_model_ = SAPD(soft=True if config.MODE in [2, 4] else False, num_cls=config.NUM_CLS,
                               depth=config.SUBNET_DEPTH, resnet=config.BACKBONE, freeze_bn=config.FREEZE_BN)

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


if __name__ == '__main__':
    print("[INFO] Initializing... ")
    _init()

    stage = f"[INFO] Stage {config.MODE} :"

    Optimizer = create_optimizer(
        opt_name=config.OPTIMIZER,
        base_lr=config.BASE_LR,
        m=config.MOMENTUM,
        decay=config.DECAY
    )

    Model_Name = create_model_name(
        info_=config.NAME,
        epochs_=config.EPOCHs,
        phi_=config.PHI,
        backbone_=config.BACKBONE,
        depth_=config.SUBNET_DEPTH,
        batch_=config.BATCH_SIZE
    )

    model, pred_model = model_compile(stage, Model_Name, Optimizer)
    print("[INFO] Testing ... Done. ")
