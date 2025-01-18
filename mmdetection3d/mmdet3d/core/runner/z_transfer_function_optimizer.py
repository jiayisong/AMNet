import math
from mmcv.runner.optimizer.builder import OPTIMIZERS
import torch
from torch.optim import _functional as F
# from torch.optim.adamw import AdamW
from torch.optim.optimizer import Optimizer
from torch.optim.sgd import SGD


@OPTIMIZERS.register_module()
class ZTransferFunctionOptimizer(Optimizer):
    def __init__(self, params, lr=1e-4, weight_decay=5e-3, numerator=(0.1,), denominator=(1, -0.9),
                 ):

        defaults = dict(lr=lr, weight_decay=weight_decay, numerator=numerator, denominator=denominator, )
        super(ZTransferFunctionOptimizer, self).__init__(params, defaults)

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
            numerator = group['numerator']
            denominator = group['denominator']
            x_n = len(numerator) - 1
            y_n = len(denominator) - 1
            for p in group['params']:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                # state = self.state[p]

                # State initialization
                if len(self.state[p]) == 0:
                    # Exponential moving average of gradient values
                    self.state[p]['x_state'] = [torch.zeros_like(p) for _ in range(x_n)]
                    self.state[p]['y_state'] = [torch.zeros_like(p) for _ in range(y_n)]
                xk = p.grad
                g = xk * numerator[0]
                for a, x_k_i in zip(numerator[1:], self.state[p]['x_state']):
                    g.add_(x_k_i, alpha=a)
                for b, y_k_i in zip(denominator[1:], self.state[p]['y_state']):
                    g.add_(y_k_i, alpha=-b)
                g = g / denominator[0]
                if x_n > 0:
                    self.state[p]['x_state'].pop(-1)
                    self.state[p]['x_state'].insert(0, xk.clone())
                if y_n > 0:
                    self.state[p]['y_state'].pop(-1)
                    self.state[p]['y_state'].insert(0, g)
                p.mul_(1 - lr * weight_decay).add_(g, alpha=-lr)
        return loss