import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as k
from tensorflow.keras.callbacks import LearningRateScheduler
from eval.voc import Evaluate


class History(tf.keras.callbacks.Callback):
    def __init__(self, lr_scheduler=None, wd_scheduler=None):
        super(History, self).__init__()
        self.lr_scheduler = lr_scheduler
        self.wd_scheduler = wd_scheduler

    def on_epoch_begin(self, epoch, logs=None):
        if self.lr_scheduler:
            k.set_value(self.model.optimizer.lr, self.lr_scheduler(epoch))

        lr = float(k.get_value(self.model.optimizer.lr))

        if self.wd_scheduler:
            try:
                k.set_value(self.model.optimizer.weight_decay, self.wd_scheduler(epoch))
                wd = float(k.get_value(self.model.optimizer.weight_decay))
                print(f"[INFO] From History Callback: EP:{epoch} LR: {lr}, WD: {wd}")
            except:
                print(f"[INFO] From History Callback: EP:{epoch} LR: {lr}")


def create_callbacks(
        config,
        pred_mod,
        test_gen,
        tensorboard=0,
        tensorboard_path=None,
):
    cbs = []
    info_ = ''

    if config.EVALUATION:
        info_ = info_ + "Evaluation, "
        cbs.append(
            Evaluate(test_gen, pred_mod)
        )

    if tensorboard:
        info_ = info_ + "Tensorboard, "

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

    if config.USING_HISTORY:
        info_ = info_ + "History, "

        _decay_dict = {
            "total_epochs": config.EPOCHs,
            "warm_up": config.USING_WARMUP,
            "warm_up_epochs": config.WP_EPOCHs,
            "warm_up_ratio": config.WP_RATIO,
            "alpha": config.ALPHA,
        }

        cbs.append(
            History(
                lr_scheduler=cosine_decay_scheduler(
                    param="LR",
                    init_val=config.BASE_LR,
                    **_decay_dict
                ),
                wd_scheduler=cosine_decay_scheduler(
                    param="WD",
                    init_val=config.DECAY,
                    **_decay_dict
                )
            )
        )

    # Early stopping.
    cbs.append(
        keras.callbacks.EarlyStopping(
            monitor='loss',
            mode='min',
            patience=3,
            restore_best_weights=True,
            verbose=1,
        )
    )
    print(info_ + 'EarlyStopping')
    return cbs


def wd_cosine_decay(epochs, initial_lr=1e-4, warm_up=0, warm_up_epochs=0, warm_up_ratio=0.1, alpha=0.):
    eps = (epochs - warm_up_epochs) - 1 if warm_up else epochs
    min_lr = warm_up_ratio * initial_lr

    def warm_up_lr(epoch):
        decay = min_lr + (initial_lr - min_lr) * (epoch / warm_up_epochs)
        print(f'-- [INFO] Warmup ({epoch + 1}/{warm_up_epochs}) WD : {decay}')
        return decay

    def cosine_decay(epoch):
        ep = (epoch - warm_up_epochs) if warm_up else epoch
        cosine = 0.5 * (1 + np.cos(np.pi * ep / eps))
        cosine = (1 - alpha) * cosine + alpha
        decay = initial_lr * cosine
        print(f'-- [INFO] Cosine-Decay WD : {decay}')
        return decay

    def lr_scheduler(epoch):
        decay = warm_up_lr(epoch) if warm_up and epoch < warm_up_epochs else cosine_decay(epoch)
        return decay

    return lr_scheduler


def lr_cosine_decay(epochs, initial_lr=1e-4, warm_up=0, warm_up_epochs=0, warm_up_ratio=0.1, alpha=0.):
    eps = (epochs - warm_up_epochs) - 1 if warm_up else epochs
    min_lr = warm_up_ratio * initial_lr

    def warm_up_lr(epoch):
        decay = min_lr + (initial_lr - min_lr) * (epoch / warm_up_epochs)
        print(f'-- [INFO] Warmup ({epoch + 1}/{warm_up_epochs}) LR : {decay}')
        return decay

    def cosine_decay(epoch):
        ep = (epoch - warm_up_epochs) if warm_up else epoch
        cosine = 0.5 * (1 + np.cos(np.pi * ep / eps))
        cosine = (1 - alpha) * cosine + alpha
        decay = initial_lr * cosine
        print(f'-- [INFO] Cosine-Decay LR : {decay}')
        return decay

    def lr_scheduler(epoch, lr):
        decay = warm_up_lr(epoch) if warm_up and epoch < warm_up_epochs else cosine_decay(epoch)
        return decay

    return LearningRateScheduler(lr_scheduler)


def cosine_decay_scheduler(
        param: str,
        total_epochs=100,
        init_val=1e-4,
        warm_up=0,
        warm_up_epochs=0,
        warm_up_ratio=0.1,
        alpha=0.,
        using_keras=0,
):
    t_eps = (total_epochs - warm_up_epochs) - 1 if warm_up else total_epochs
    min_val = warm_up_ratio * init_val

    def _warm_up(current_epoch):
        decay = min_val + (init_val - min_val) * (current_epoch / warm_up_epochs)
        print(f'-- [INFO] Warmup ({current_epoch + 1}/{warm_up_epochs}) {param} : {decay}')
        return decay

    def _cosine_decay(current_epoch):
        c_ep = (current_epoch - warm_up_epochs) if warm_up else current_epoch
        cosine = 0.5 * (1 + np.cos(np.pi * c_ep / t_eps))
        cosine = (1 - alpha) * cosine + alpha
        decay = init_val * cosine
        print(f'-- [INFO] Cosine-Decay {param} : {decay}')
        return decay

    def _output(epoch):
        return _warm_up(epoch) if warm_up and epoch < warm_up_epochs else _cosine_decay(epoch)

    if using_keras:
        def _scheduler(epoch, lr):
            return _output(epoch)

        return LearningRateScheduler(_scheduler)

    else:
        def _scheduler(epoch):
            return _output(epoch)

        return _scheduler
