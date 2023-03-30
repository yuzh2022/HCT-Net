"""Learning Rate Schedulers"""
from __future__ import division
from math import pi, cos

class LRScheduler(object):
    r"""Learning Rate Scheduler
    For mode='step', we multiply lr with `decay_factor` at each epoch in `step`.
    For mode='poly'::
        lr = targetlr + (baselr - targetlr) * (1 - iter / maxiter) ^ power
    For mode='cosine'::
        lr = targetlr + (baselr - targetlr) * (1 + cos(pi * iter / maxiter)) / 2
    If warmup_epochs > 0, a warmup stage will be inserted before the main lr scheduler.
    For warmup_mode='linear'::
        lr = warmup_lr + (baselr - warmup_lr) * iter / max_warmup_iter
    For warmup_mode='constant'::
        lr = warmup_lr
    Parameters
    ----------
    mode : str
        Modes for learning rate scheduler.
        Currently it supports 'step', 'poly' and 'cosine'.
    niters : int
        Number of iterations in each epoch.
    base_lr : float
        Base learning rate, i.e. the starting learning rate.
    epochs : int
        Number of training epochs.
    step : list
        A list of epochs to decay the learning rate.
    decay_factor : float
        Learning rate decay factor.
    targetlr : float
        Target learning rate for poly and cosine, as the ending learning rate.
    power : float
        Power of poly function.
    warmup_epochs : int
        Number of epochs for the warmup stage.
    warmup_lr : float
        The base learning rate for the warmup stage.
    warmup_mode : str
        Modes for the warmup stage.
        Currently it supports 'linear' and 'constant'.
    """
    def __init__(self, optimizer, niters):
        #super(LRScheduler, self).__init__()

        self.mode = 'cosine'
        self.warmup_mode = 'linear'
        assert(self.mode in ['step', 'poly', 'cosine'])
        assert(self.warmup_mode in ['linear', 'constant'])

        self.optimizer = optimizer

        self.base_lr = 0.5
        self.learning_rate = self.base_lr
        self.niters = niters

        self.step = [30, 60, 90]
        self.decay_factor = 0.1
        self.targetlr =0.0
        self.power = 1
        self.warmup_lr = 0.0001
        self.max_iter = 80 * niters
        self.warmup_iters = 5 * niters

    def update(self, i, epoch):
        T = epoch * self.niters + i
        assert (T >= 0 and T <= self.max_iter)

        if self.warmup_iters > T:
            # Warm-up Stage
            if self.warmup_mode == 'linear':
                self.learning_rate = self.warmup_lr + (self.base_lr - self.warmup_lr) * \
                    T / self.warmup_iters
            elif self.warmup_mode == 'constant':
                self.learning_rate = self.warmup_lr
            else:
                raise NotImplementedError
        else:
            if self.mode == 'step':
                count = sum([1 for s in self.step if s <= epoch])
                self.learning_rate = self.base_lr * pow(self.decay_factor, count)
            elif self.mode == 'poly':
                self.learning_rate = self.targetlr + (self.base_lr - self.targetlr) * \
                    pow(1 - (T - self.warmup_iters) / (self.max_iter - self.warmup_iters), self.power)
            elif self.mode == 'cosine':
                self.learning_rate = self.targetlr + (self.base_lr - self.targetlr) * \
                    (1 + cos(pi * (T - self.warmup_iters) / (self.max_iter - self.warmup_iters))) / 2
            else:
                raise NotImplementedError

        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = self.learning_rate
