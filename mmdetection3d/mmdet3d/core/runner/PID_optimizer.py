import math
from mmcv.runner.optimizer.builder import OPTIMIZERS
import torch
from torch.optim import _functional as F
# from torch.optim.adamw import AdamW
from torch.optim.optimizer import Optimizer
from torch.optim.sgd import SGD


@OPTIMIZERS.register_module()
class PID_SGD(Optimizer):
    r"""Implements AdamW algorithm.

    The original Adam algorithm was proposed in `Adam: A Method for Stochastic Optimization`_.
    The AdamW variant was proposed in `Decoupled Weight Decay Regularization`_.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, pid=(0., 1, 0.), weight_decay=1e-2, incomplete_differential=None, integral_separation=None):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, pid=pid, weight_decay=weight_decay)
        self.incomplete_differential = incomplete_differential
        self.integral_separation = integral_separation
        super(PID_SGD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            P, I, D = group['pid']
            for p in group['params']:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                grad = p.grad
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state['accumulation_grad'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['last_grad'] = p.grad.clone()
                    state['d_filter'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                last_grad = state['last_grad']
                accumulation_grad = state['accumulation_grad']
                if self.incomplete_differential is not None:
                    d_filter = state['d_filter']
                    u_d = grad - last_grad
                    d_filter.mul_(self.incomplete_differential).add_(u_d, alpha=1 - self.incomplete_differential)
                else:
                    d_filter = grad - last_grad
                accumulation_grad.add_(self.f(grad))
                last_grad.mul_(0).add_(grad)
                g = P * grad + I * accumulation_grad + D * d_filter
                p.mul_(1 - lr * weight_decay).add_(g, alpha=-lr)
        return loss

    def f(self, e):
        if self.integral_separation is not None:
            A, B = self.integral_separation
            e_abs = e.abs()
            C1 = e_abs <= B
            C2 = ~C1 & (e_abs <= (A + B))
            #C3 = ~(C1 | C2)
            return e * (C1 + C2 * (A - e_abs + B)/A)
        else:
            return e
