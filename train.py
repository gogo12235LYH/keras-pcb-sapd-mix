import tensorflow as tf
from tensorflow.python.keras.utils.data_utils import get_file
from generators.voc import PascalVocGenerator
from preprocess.color_aug import VisualEffect
from preprocess.misc_aug import MiscEffect
import tensorflow.keras as keras
import tensorflow.keras.backend as k
from eval.voc import Evaluate
from models import SAPD
from tensorflow.keras.callbacks import LearningRateScheduler
import numpy as np
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


def create_generators(batch_size=2,
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

    train_generator_ = PascalVocGenerator(
        path,
        'trainval' if config.MixUp_AUG == 0 else 'trainval_mixup',
        skip_difficult=True,
        misc_effect=misc_effect,
        visual_effect=visual_effect,
        **common_args
    )

    validation_generator_ = PascalVocGenerator(
        path,
        'val',
        skip_difficult=True,
        shuffle_groups=False,
        **common_args
    )

    test_generator_ = PascalVocGenerator(
        path,
        'test',
        skip_difficult=True,
        shuffle_groups=False,
        **common_args
    )

    return train_generator_, validation_generator_, test_generator_


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


class History(tf.keras.callbacks.Callback):
    def __init__(self, wd_scheduler):
        super(History, self).__init__()
        self.lr = []
        self.wd = []
        self.wd_scheduler = wd_scheduler

    def on_train_begin(self, logs=None):
        self.lr = []
        self.wd = []

    def on_epoch_begin(self, epoch, logs=None):
        lr = float(k.get_value(self.model.optimizer.lr))
        k.set_value(self.model.optimizer.weight_decay, self.wd_scheduler(epoch))
        wd = float(k.get_value(self.model.optimizer.weight_decay))
        print(f"[INFO] From History Callback: EP:{epoch} LR: {lr}, WD: {wd}")

    def on_epoch_end(self, epoch, logs=None):
        self.lr.append(float(k.get_value(self.model.optimizer.lr)))
        self.wd.append(float(k.get_value(self.model.optimizer.weight_decay)))


def create_callbacks(pred_mod, test_gen, m_eval=0, tensorboard=0, tensorboard_path=None):
    cbs = []
    info_ = ''

    if m_eval:
        info_ = info_ + "Evaluation, "
        cbs.append(
            Evaluate(test_gen, pred_mod)
        )

    if tensorboard:
        info_ = info_ + "Tensorboard "
        # print("Tensorboard, ", end='')
        cbs.append(
            keras.callbacks.TensorBoard(
                log_dir='{file_path}/logs'.format(file_path=tensorboard_path),
                histogram_freq=0,
                write_graph=True,
                write_images=False,
                profile_batch=config.BATCH_SIZE,
                embeddings_freq=0,
                embeddings_metadata=None
            )
        )

    if config.LR_Scheduler == 0:
        print(info_)
        return cbs

    if config.LR_Scheduler == 1:
        if config.USING_WARMUP:
            info_ = info_ + "Cosine-Decay + Warm-Up, "
            # print("Cosine Decay + Warm Up, ", end='' if config.USING_HISTORY else None)
            cbs.append(
                cosine_decay3(
                    initial_lr=config.BASE_LR,
                    epochs=config.EPOCHs,
                    warm_up=1,
                    warm_up_epochs=config.WP_EPOCHs,
                    alpha=config.ALPHA
                )
            )
        else:
            info_ = info_ + "Cosine-Decay, "
            # print("Cosine Decay, ", end='' if config.USING_HISTORY else None)
            cbs.append(
                cosine_decay3(
                    initial_lr=config.BASE_LR,
                    epochs=config.EPOCHs,
                    warm_up=0,
                    warm_up_epochs=0,
                    alpha=config.ALPHA

                )
            )

    elif config.LR_Scheduler == 2:
        info_ = info_ + "Cosine-Decay(RS), "
        # print("Cosine Decay(RS), ", end='' if config.USING_HISTORY else None)
        cbs.append(
            cosine_decay_rs(
                initial_lr=config.BASE_LR,
                epochs=config.EPOCHs,
                epoch_r=config.EPOCHs_RESTART,
                alpha=config.ALPHA
            )
        )

    if config.USING_HISTORY == 1:
        info_ = info_ + "History"
        # print("History")
        h = History(
            wd_scheduler=wd_cosine_decay3(
                initial_lr=config.DECAY,
                epochs=config.EPOCHs,
                warm_up=config.USING_WARMUP,
                warm_up_epochs=config.WP_EPOCHs,
                alpha=config.ALPHA
            )
        )
        cbs.append(h)

    elif config.USING_HISTORY == 2:
        info_ = info_ + "History(RS)"
        # print("History(RS)")
        h = History(
            wd_scheduler=wd_cosine_decay_rs(
                initial_lr=config.DECAY,
                epochs=config.EPOCHs,
                epoch_r=config.EPOCHs_RESTART,
                alpha=config.ALPHA
            )
        )
        cbs.append(h)

    cbs.append(
        keras.callbacks.EarlyStopping(
            monitor='loss',
            mode='min',
            patience=3,
            restore_best_weights=True,
            verbose=1,
        )
    )
    print(info_ + 'EarlyStopping,')
    return cbs


def wd_cosine_decay3(epochs, initial_lr=1e-5, warm_up=0, warm_up_epochs=0, alpha=0.):
    eps = (epochs - warm_up_epochs) - 1 if warm_up else epochs
    min_lr = config.WP_RATIO * initial_lr

    def warm_up_lr(epoch):
        decay = min_lr + (initial_lr - min_lr) * (epoch / warm_up_epochs)
        print(f'-- [INFO] Warmup ({epoch + 1}/{warm_up_epochs}) WD : {decay}')
        return decay

    def cosine_decay_lr(epoch):
        ep = (epoch - warm_up_epochs) if warm_up else epoch
        cosine = 0.5 * (1 + np.cos(np.pi * ep / eps))
        cosine = (1 - alpha) * cosine + alpha
        decay = initial_lr * cosine
        print(f'-- [INFO] Cosine-Decay WD : {decay}')
        return decay

    def lr_scheduler(epoch):
        decay = warm_up_lr(epoch) if warm_up and epoch < warm_up_epochs else cosine_decay_lr(epoch)
        return decay

    return lr_scheduler


def cosine_decay3(epochs, initial_lr=1e-4, warm_up=1, warm_up_epochs=5, alpha=0.):
    eps = (epochs - warm_up_epochs) - 1 if warm_up else epochs
    min_lr = config.WP_RATIO * initial_lr

    def warm_up_lr(epoch):
        decay = min_lr + (initial_lr - min_lr) * (epoch / warm_up_epochs)
        print(f'-- [INFO] Warmup ({epoch + 1}/{warm_up_epochs}) LR : {decay}')
        return decay

    def cosine_decay_lr(epoch):
        ep = (epoch - warm_up_epochs) if warm_up else epoch
        cosine = 0.5 * (1 + np.cos(np.pi * ep / eps))
        cosine = (1 - alpha) * cosine + alpha
        decay = initial_lr * cosine
        print(f'-- [INFO] Cosine-Decay LR : {decay}')
        return decay

    def lr_scheduler(epoch, lr):
        decay = warm_up_lr(epoch) if warm_up and epoch < warm_up_epochs else cosine_decay_lr(epoch)
        return decay

    return LearningRateScheduler(lr_scheduler)


def cosine_decay_rs(epochs, epoch_r=25, initial_lr=1e-4, alpha=0.02):
    def cosine_decay_lr(epoch):

        if epoch >= epoch_r:
            eps = epoch_r * 3
            ep = epoch - epoch_r
            lr = initial_lr * config.RS_RATIO

        else:
            ep = epoch
            eps = epoch_r
            lr = initial_lr

        cosine = 0.5 * (1 + np.cos(np.pi * ep / eps))
        cosine = (1 - alpha) * cosine + alpha
        decay = lr * cosine
        print(f'-- [INFO] Cosine-Decay LR : {decay}')
        return decay

    def lr_scheduler(epoch, lr):
        decay = cosine_decay_lr(epoch)
        return decay

    return LearningRateScheduler(lr_scheduler)


def wd_cosine_decay_rs(epochs, epoch_r=25, initial_lr=1e-4, alpha=0.02):
    def cosine_decay_lr(epoch):

        if epoch >= epoch_r:
            eps = epoch_r * 3
            ep = epoch - epoch_r
            lr = initial_lr * config.RS_RATIO

        else:
            ep = epoch
            eps = epoch_r
            lr = initial_lr

        cosine = 0.5 * (1 + np.cos(np.pi * ep / eps))
        cosine = (1 - alpha) * cosine + alpha
        decay = lr * cosine
        print(f'-- [INFO] Cosine-Decay WD : {decay}')
        return decay

    def lr_scheduler(epoch):
        decay = cosine_decay_lr(epoch)
        return decay

    return lr_scheduler


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

    """ Output Weight Name """
    Model_Name = create_model_name(
        info_=config.NAME,
        epochs_=config.EPOCHs,
        phi_=config.PHI,
        backbone_=config.BACKBONE,
        depth_=config.SUBNET_DEPTH,
        batch_=config.BATCH_SIZE
    )

    print(f"{stage} Creating Generators... ")
    train_generator, val_generator, test_generator = create_generators(
        batch_size=config.BATCH_SIZE,
        phi=config.PHI,
        misc_aug=config.MISC_AUG,
        visual_aug=config.VISUAL_AUG,
        path=config.DATABASE_PATH,
    )

    """ Multi GPU Accelerating"""
    if config.MULTI_GPU:
        mirrored_strategy = tf.distribute.MirroredStrategy()
        with mirrored_strategy.scope():
            model, pred_model = model_compile(stage, Model_Name, Optimizer)
    else:
        model, pred_model = model_compile(stage, Model_Name, Optimizer)

    print(f"{stage} Model Name : {Model_Name}")

    print(f"{stage} Creating Callbacks... ", end='')
    callbacks = create_callbacks(
        pred_mod=pred_model,
        test_gen=test_generator,
        m_eval=config.EVALUATION,
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
    save_model_name = Model_Name + "-soft" if config.MODE == 2 else Model_Name
    print(f"{stage} Saving Model Weights : {save_model_name}.h5 ... ")
    model.save_weights(f'{save_model_name}.h5')


if __name__ == '__main__':
    main()
