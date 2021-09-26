from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K
import numpy as np


class CDScheduler(Callback):
    """
        Cosine Decay
    """

    def __init__(
            self,
            base_lr,
            alpha=0.0,
            initial_epoch=0,
            warm_up=False,
            warm_up_iterations=0,
            save_txt=False,
            stage=0,
    ):
        super(CDScheduler, self).__init__()
        self.base_lr = base_lr
        self.alpha = alpha
        self.initial_epoch = initial_epoch
        self.warm_up = warm_up
        self.warm_up_iterations = warm_up_iterations
        self.save_txt = save_txt
        self.stage = stage

        self.total_iteration = 0
        self.lr = 0.
        self.losses = []
        self.learning_rates = []
        self.iteration = 0

    def on_train_begin(self, logs=None):
        p = self.params
        self.total_iteration = p['steps'] * p['epochs']
        self.iteration = self.initial_epoch * p['steps']
        print(f"[INFO] Total Iterations : {self.total_iteration}")
        print(f"[INFO] Epochs : {p['epochs']}")
        print(f"[INFO] Steps per epoch : {p['steps']}")

        self.lr = cosine_decay_with_warm_up(
            iteration=self.iteration,
            iterations=self.total_iteration,
            initial_lr=self.base_lr,
            warm_up=self.warm_up,
            warm_up_epochs=self.warm_up_iterations,
            alpha=self.alpha
        )
        K.set_value(self.model.optimizer.lr, self.lr)

    def on_epoch_begin(self, epoch, logs=None):
        print(f"[INFO] Current Iteration : {self.iteration + 1}")
        print(f"[INFO] Learning Rate : {self.lr}")
        self.learning_rates.append(self.lr)

    def on_batch_end(self, batch, logs=None):
        self.iteration += 1

        # Get Loss and store it.
        loss = logs.get('loss')
        self.losses.append(loss)

        self.lr = cosine_decay_with_warm_up(
            iteration=self.iteration,
            iterations=self.total_iteration,
            initial_lr=self.base_lr,
            warm_up=self.warm_up,
            warm_up_epochs=self.warm_up_iterations,
            alpha=self.alpha
        )
        K.set_value(self.model.optimizer.lr, self.lr)

    def on_train_end(self, logs=None):
        if self.save_txt:
            np.savetxt(f'losses_Stage{self.stage}.txt', self.losses)
            np.savetxt(f'learning_Stage{self.stage}.txt', self.learning_rates)


def cosine_decay_with_warm_up(iteration, iterations, initial_lr=1., warm_up=1, warm_up_epochs=2, alpha=0.):
    eps = (iterations - warm_up_epochs) - 1 if warm_up else iterations
    min_lr = (0.5 * (1 + np.cos(np.pi * (eps - 0) / eps)) * (1 - alpha) + alpha) * initial_lr

    def warm_up_lr(epoch):
        decay = min_lr + (initial_lr - min_lr) * (epoch / (warm_up_epochs + 0))
        return decay

    def cosine_decay_lr(epoch):
        ep = (epoch - warm_up_epochs) if warm_up else epoch
        cosine = 0.5 * (1 + np.cos(np.pi * ep / eps))
        cosine = (1 - alpha) * cosine + alpha
        decay = initial_lr * cosine
        return decay

    def lr_scheduler(epoch):
        decay = warm_up_lr(epoch) if warm_up and epoch < warm_up_epochs else cosine_decay_lr(epoch)
        return decay

    return lr_scheduler(epoch=iteration)


class CDCLRScheduler(Callback):
    """
        Cosine Decay with Cycle Restart
    """

    def __init__(
            self,
            base_lr,
            alpha=0.,
            epochs=None,
            initial_epoch=0,
            steps_per_epoch=None,
            cycle_length=8,
            factor=1
    ):
        super(CDCLRScheduler, self).__init__()
        self.base_lr = base_lr
        self.alpha = alpha

        self.epochs = epochs
        self.initial_epoch = initial_epoch
        self.batch_since_restart = 0
        self.next_restart = cycle_length

        self.iteration = 0

        self.steps_per_epoch = steps_per_epoch

        self.cycle_length = cycle_length
        self.factor = factor

    def clr(self):
        fraction_to_restart = self.batch_since_restart / (self.steps_per_epoch * self.cycle_length)
        decay = 0.5 * (1 + np.cos(fraction_to_restart * np.pi))
        decay = self.alpha + (1 - self.alpha) * decay
        lr = self.base_lr * decay
        return lr

    def on_train_begin(self, logs=None):
        p = self.params

        if self.steps_per_epoch is None:
            self.steps_per_epoch = p['steps']

        if self.epochs is None:
            self.epochs = p['epochs']

        print(f"[INFO] Epochs : {p['epochs']}")
        print(f"[INFO] Steps per epoch : {p['steps']}")
        print(f"[INFO] Total Iterations : {p['epochs'] * p['steps']}")

        if self.initial_epoch != 0:
            for i in range(self.initial_epoch):
                self.batch_since_restart += self.steps_per_epoch
                self.iteration += self.steps_per_epoch

                if i + 1 == self.next_restart:
                    self.batch_since_restart = 0
                    self.cycle_length = np.ceil(self.cycle_length * self.factor)
                    self.next_restart += self.cycle_length

        print(f"[INFO] Current Iteration : {self.iteration + 1}")
        K.set_value(self.model.optimizer.lr, self.clr())

    def on_batch_end(self, batch, logs=None):
        self.batch_since_restart += 1
        K.set_value(self.model.optimizer.lr, self.clr())

    def on_epoch_begin(self, epoch, logs=None):
        print(f"[INFO] Learning Rate : {self.clr()}")

    def on_epoch_end(self, epoch, logs=None):
        if epoch + 1 == self.next_restart:
            self.batch_since_restart = 0
            self.cycle_length = np.ceil(self.cycle_length * self.factor)
            self.next_restart += self.cycle_length


class CDCLRScheduler2(Callback):
    """
        Cosine Decay with Cycle Restart
    """

    def __init__(
            self,
            base_lr,
            alpha=0.,
            initial_epoch=0,
            steps_per_epoch=None,
            cycle_length=8,
            factor=1,
            warm_up=False,
            warm_up_epochs=0,
            warm_up_lr_factor=0.1,
            save_txt=True,
            file_name=None,
    ):
        super(CDCLRScheduler2, self).__init__()
        self.base_lr = base_lr
        self.alpha = alpha
        self.initial_epoch = initial_epoch
        self.steps_per_epoch = steps_per_epoch
        self.cycle_length = cycle_length
        self.factor = factor
        self.warm_up = warm_up
        self.warm_up_epochs = warm_up_epochs
        self.warm_up_lr = base_lr * warm_up_lr_factor
        self.save_txt = save_txt
        self.file_name = file_name

        self.total_steps = 0
        self.total_wp_steps = 0
        self.batch_since_restart = 0
        self.step = 0
        self.next_restart = cycle_length
        self.flag = 1

        self.learning_rates = []
        self.losses = []

    def clr(self):

        if self.warm_up and self.step == self.total_wp_steps:
            self.batch_since_restart = 0

        length = (self.steps_per_epoch * self.cycle_length)

        if self.warm_up and self.step > self.total_wp_steps and self.flag:
            if self.step < self.steps_per_epoch * self.cycle_length:
                length -= (self.steps_per_epoch * self.warm_up_epochs)

            else:
                self.flag = 0

        fraction_to_restart = self.batch_since_restart / length
        decay = 0.5 * (1 + np.cos(fraction_to_restart * np.pi))
        decay = self.alpha + (1 - self.alpha) * decay
        lr = self.base_lr * decay
        return lr

    def wlr(self):
        decay = self.warm_up_lr + (self.base_lr - self.warm_up_lr) * (
                self.step / (self.steps_per_epoch * self.warm_up_epochs))
        return decay

    def check_restart(self, epoch):
        if epoch + 1 == self.next_restart:
            self.batch_since_restart = 0
            self.cycle_length = np.ceil(self.cycle_length * self.factor)
            self.next_restart += self.cycle_length

    def batch_count(self, num_steps):
        self.batch_since_restart += num_steps
        self.step += num_steps

    def on_train_begin(self, logs=None):
        p = self.params

        if self.steps_per_epoch is None:
            self.steps_per_epoch = p['steps']

        self.total_steps = p['epochs'] * p['steps']

        if self.warm_up:
            self.total_wp_steps = self.warm_up_epochs * self.steps_per_epoch

        # check for initial epoch
        if self.initial_epoch != 0:
            for i in range(self.initial_epoch):
                self.batch_count(num_steps=self.steps_per_epoch)
                self.check_restart(epoch=i)

            if self.warm_up and self.step > self.total_wp_steps:
                self.batch_since_restart -= self.total_wp_steps

        # if warm up is used, and initial epoch must be 0.
        if self.warm_up and self.initial_epoch == 0:
            K.set_value(self.model.optimizer.lr, self.wlr())

        else:
            K.set_value(self.model.optimizer.lr, self.clr())

        if self.save_txt and self.file_name is not None:
            self.file_name = 'CDCLRScheduler'

        print(f"[INFO] Epochs : {p['epochs']}")
        print(f"[INFO] Steps per epoch : {p['steps']}")
        print(f"[INFO] Total steps : {self.total_steps}")
        print(f"[INFO] Current Step : {self.step + 1}")

    def on_batch_end(self, batch, logs=None):
        if self.warm_up and self.step < self.steps_per_epoch * self.warm_up_epochs:
            self.learning_rates.append(self.wlr())

        else:
            self.learning_rates.append(self.clr())

        self.losses.append(logs.get("loss"))
        self.batch_count(num_steps=1)

        if self.warm_up and self.step < self.steps_per_epoch * self.warm_up_epochs:
            K.set_value(self.model.optimizer.lr, self.wlr())

        else:
            K.set_value(self.model.optimizer.lr, self.clr())

    def on_epoch_begin(self, epoch, logs=None):
        if self.warm_up and self.step < self.steps_per_epoch * self.warm_up_epochs:
            print(f"[INFO] Steps {self.step + 1}/{self.total_wp_steps} | Warm Up Lr : {self.wlr()}")

        else:
            print(f"[INFO] Steps {self.step + 1}/{self.total_steps} | Cosine Decay Lr : {self.clr()}")

    def on_epoch_end(self, epoch, logs=None):
        self.check_restart(epoch=epoch)

    def on_train_end(self, logs=None):
        if self.save_txt:
            np.savetxt(f'{self.file_name}_lrs.txt', self.learning_rates)
            np.savetxt(f'{self.file_name}_losses.txt', self.losses)
