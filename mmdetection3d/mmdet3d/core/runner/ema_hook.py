from mmdet.core.hook.ema import BaseEMAHook
from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class LrReciprocalMomentumEMAHook(BaseEMAHook):
    """EMAHook using linear momentum strategy.

    Args:
        warm_up (int): During first warm_up steps, we may use smaller decay
            to update ema parameters more slowly. Defaults to 100.
    """

    def get_momentum(self, runner):
        lrs = runner.current_lr()
        if isinstance(lrs, dict):
            for name, value in lrs.items():
                lr = value[0]
        else:
            lr = lrs[0]
        momentum = self.momentum / lr
        # print(momentum)
        return momentum


@HOOKS.register_module()
class LrMomentumEMAHook(BaseEMAHook):
    """EMAHook using linear momentum strategy.

    Args:
        warm_up (int): During first warm_up steps, we may use smaller decay
            to update ema parameters more slowly. Defaults to 100.
    """

    def __init__(self,
                 momentum=0.0002,
                 interval=1,
                 skip_buffers=False,
                 resume_from=None,
                 momentum_fun=None):
        self.momentum = momentum
        self.skip_buffers = skip_buffers
        self.interval = interval
        self.checkpoint = resume_from
        self.momentum_fun = momentum_fun

    def get_momentum(self, runner):
        lrs = runner.current_lr()
        if isinstance(lrs, dict):
            for name, value in lrs.items():
                lr = value[0]
        else:
            lr = lrs[0]
        momentum = self.momentum * lr
        # print(momentum)
        return momentum


@HOOKS.register_module()
class MeanFilterEMAHook(BaseEMAHook):
    """EMAHook using linear momentum strategy.

    Args:
        warm_up (int): During first warm_up steps, we may use smaller decay
            to update ema parameters more slowly. Defaults to 100.
    """

    def __init__(self,
                 start_epoch,
                 interval=1,
                 skip_buffers=False,
                 resume_from=None, ):
        self.start_epoch = start_epoch
        self.skip_buffers = skip_buffers
        self.interval = interval
        self.checkpoint = resume_from

    def get_momentum(self, runner):
        if runner.epoch >= self.start_epoch:
            x = runner.iter - self.start_epoch * len(runner.data_loader)
            return 1 / (x + 1)
        else:
            return 1


@HOOKS.register_module()
class MyLinearMomentumEMAHook(BaseEMAHook):
    """EMAHook using linear momentum strategy.

    Args:
        warm_up (int): During first warm_up steps, we may use smaller decay
            to update ema parameters more slowly. Defaults to 100.
    """
    def __init__(self, warm_up=10, **kwargs):
        super(MyLinearMomentumEMAHook, self).__init__(**kwargs)
        self.momentum_fun = lambda x: max(1 - (1 - self.momentum) ** self.interval,
                                          (warm_up - 1) / (warm_up + x))
