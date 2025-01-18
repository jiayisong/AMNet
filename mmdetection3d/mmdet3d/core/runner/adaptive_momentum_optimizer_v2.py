import math
from mmcv.runner.optimizer.builder import OPTIMIZERS
import torch
from torch.optim import _functional as F
# from torch.optim.adamw import AdamW
from torch.optim.optimizer import Optimizer
from torch.optim.sgd import SGD


@OPTIMIZERS.register_module()
class AdaptiveMomentumOptimizerv2(Optimizer):
    def __init__(self, params, lr=1e-4, weight_decay=5e-3, momentum_beta=10):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=1, momentum_beta=momentum_beta)
        super(AdaptiveMomentumOptimizerv2, self).__init__(params, defaults)

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
        norms = []
        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            momentum_beta = group['momentum_beta']
            for p in group['params']:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                # state = self.state[p]
                state = self.state[p]
                grad = p.grad  # + weight_decay * p
                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    # state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['momentum_sum'] = 0
                # state['step'] += 1
                exp_avg = state['exp_avg']
                momentum_sum = state['momentum_sum']
                exp_avg.add_(grad)
                momentum_sum += 1
                norm = (exp_avg / momentum_sum + p * weight_decay) * (-lr)
                p.add_(norm)
                norms.append(norm)
        total_norm = torch.norm(torch.stack([torch.norm(p, 2) for p in norms]), 2)
        adaptivemomentum = (1 - (total_norm / momentum_beta)).clamp(0, 1).item()
        for group in self.param_groups:
            group['momentum'] = 1 - 1/momentum_sum
            for p in group['params']:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                state = self.state[p]
                exp_avg = state['exp_avg']
                exp_avg.mul_(adaptivemomentum)
                momentum_sum = state['momentum_sum']
                momentum_sum *= adaptivemomentum
        return loss
