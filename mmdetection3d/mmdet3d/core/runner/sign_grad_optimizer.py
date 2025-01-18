import math
from mmcv.runner.optimizer.builder import OPTIMIZERS
import torch
from torch.optim import _functional as F
# from torch.optim.adamw import AdamW
from torch.optim.optimizer import Optimizer
from torch.optim.sgd import SGD


@OPTIMIZERS.register_module()
class SignGradOptimizer(Optimizer):
    def __init__(self, params, lr=1e-4, weight_decay=5e-3, momentum=0.9, beta=(10, 1)):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, beta=beta)
        super(SignGradOptimizer, self).__init__(params, defaults)

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
            momentum = group['momentum']
            beta1, beta2 = group['beta']
            for p in group['params']:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                # state = self.state[p]
                state = self.state[p]
                grad = p.grad
                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)
                # grad = ((p.grad * beta1 * 2 / beta2).sigmoid() - 0.5) * (beta2 * 2)
                exp_avg = state['exp_avg']
                # grad = self.nonlinear(grad, beta1, beta2)
                exp_avg.mul_(momentum).add_(grad, alpha=1 - momentum)
                exp_avg = self.nonlinear(exp_avg, beta1, beta2)
                # grad = ((exp_avg * beta1 * 2 / beta2).sigmoid() - 0.5) * (beta2 * 2)
                p.mul_(1 - lr * weight_decay).add_(exp_avg, alpha=-lr)
        return loss

    def nonlinear(self, g, beta1, beta2):
        # return ((g * beta1 * 2 / beta2).sigmoid() - 0.5) * (beta2 * 2)
        return (g * beta1).clamp(-beta2, beta2)
        # return beta2 * g.sign() * g.abs().pow(beta1)
