import math
from mmcv.runner.optimizer.builder import OPTIMIZERS
import torch
from torch.optim import _functional as F
from torch.optim.adamw import AdamW
from torch.optim.optimizer import Optimizer
from torch.optim.sgd import SGD




@OPTIMIZERS.register_module()
class AdamWAMOptimizer(Optimizer):
    def __init__(self, params, lr=1e-4, weight_decay=5e-3, g_max=9.8, alpha=1, beta=0.999, eps=1e-8,):
        defaults = dict(lr=lr, weight_decay=weight_decay, g_max=g_max, alpha=alpha,  eps=eps, beta=beta)
        super(AdamWAMOptimizer, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, grad_norm):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        total_norm = grad_norm ** 2
        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            alpha = group['alpha']
            g_max = group['g_max']
            beta = group['beta']
            for p in group['params']:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                state = self.state[p]
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state['step'] = 0
                    state['v'] = torch.zeros_like(p)
                    state['v2'] = torch.zeros_like(p)
                t = lr
                grad = p.grad
                if g_max > 0:
                    grad_square_sum = total_norm * beta * beta / 4 / g_max / g_max
                    a_g = - grad / (1 + grad_square_sum)  # ** 0.5
                else:
                    a_g = - grad
                v = state['v']
                a = a_g
                v.mul_(1 - (alpha * t * v.abs()).clamp_max(1)).add_(a, alpha=t)
                p.mul_(1 - weight_decay * lr).add_(v, alpha=lr)
                # p.add_(v, alpha=t)
                # print(p.abs().mean())
        # self.state['v'] *= (1 - min((alpha * t * abs(self.state['v'])), 1))
        # self.state['v'] += (-g * t * grad_square_sum / (1 + grad_square_sum))
        return None

    def compute_dot(self):
        dot = self.state['v']
        for group in self.param_groups:
            beta = group['beta']
            for p in group['params']:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                # state = self.state[p]
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state['v'] = torch.zeros_like(p)
                    # state['x'] = p.clone().detach()
                v = state['v']
                grad = p.grad
                dot = dot - (v * grad).sum().item() * beta
        return dot

    def compute_v_norm(self):
        v = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                # state = self.state[p]
                state = self.state[p]
                v.append(state['v'])
        temp = torch.norm(torch.stack([torch.norm(p, 2) for p in v]), 2).item()
        # print(temp, self.state['v'])
        v_norm = (self.state['v'] ** 2 + temp ** 2) ** 0.5
        return v_norm

    def transform_v(self, total_norm, dot):
        for group in self.param_groups:
            beta = group['beta']
            for p in group['params']:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                state = self.state[p]
                grad = p.grad * beta
                grad_square_sum = total_norm * beta * beta
                v = state['v']
                v.add_(grad, alpha=dot / (1 + grad_square_sum))
        self.state['v'] -= dot / (1 + grad_square_sum)
        # print('dot/1+norm', (dot / (1 + grad_square_sum)) ** 2 * (1 + grad_square_sum))

    def zoom_v(self, a):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                state = self.state[p]
                v = state['v']
                v.mul_(a)
        self.state['v'] *= a
