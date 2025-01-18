import math
from mmcv.runner.optimizer.builder import OPTIMIZERS
import torch
from torch.optim import _functional as F
from torch.optim.optimizer import Optimizer
from torch.optim.sgd import SGD
from mmcv.utils import print_log

'''
@OPTIMIZERS.register_module()
class AdaptiveMomentumOptimizer(Optimizer):
    def __init__(self, params, lr=1e-4, weight_decay=5e-3, momentum=0.9, momentum_beta=10, momentum_max=100,
                 abs_beta=(0.99, 1000)):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, momentum_beta=momentum_beta,
                        momentum_max=momentum_max, abs_beta=abs_beta)
        super(AdaptiveMomentumOptimizer, self).__init__(params, defaults)

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
            momentum_max = group['momentum_max']
            beta1, beta2 = group['abs_beta']
            momentum_beta = group['momentum_beta']
            for p in group['params']:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                # state = self.state[p]
                state = self.state[p]
                grad = p.grad# + weight_decay * p
                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    #state['momentum_sum'] = torch.zeros_like(p)
                    # state['grad_abs'] = torch.zeros_like(p)
                state['step'] += 1
                # bias_correction1 = 1 - beta1 ** state['step']
                # grad_abs = state['grad_abs']
                # grad_abs.mul_(beta1).add_(grad.abs(), alpha=1-beta1).div_(bias_correction1)
                exp_avg = state['exp_avg']
                #momentum_sum = state['momentum_sum']
                # adaptivemomentum = (1 - (exp_avg.abs() / momentum_beta)).clamp(0,  1)  # * (grad.abs() / 0.1).clamp_max(1)
                # print(adaptivemomentum.mean())
                # adaptivemomentum
                # adaptivemomentum = adaptivemomentum / (adaptivemomentum + 1 - momentum)
                # momentum_sum.add_(1).mul_(adaptivemomentum)
                # print(momentum_sum.max())
                # temp = (momentum_max / momentum_sum).clamp_max(1)
                # temp = momentum_max > momentum_sum
                # adaptivemomentum.mul_(temp)
                # momentum_sum.mul_(temp)
                # exp_avg.mul_(adaptivemomentum)
                # print(adaptivemomentum.mean())
                # exp_avg.mul_(beta1*2/beta2).sigmoid_().add_(-0.5).mul_(beta2 * 2)
                exp_avg.add_(grad, alpha=0.1)
                #momentum_sum.add_(0.1)
                # g_and_wd = exp_avg + p * weight_decay
                # exp_avg.add_(grad * 0.1 * (adaptivemomentum + 0.5))
                p.mul_(1 - lr * weight_decay)
                p.add_(exp_avg, alpha=-lr)
                adaptivemomentum = (1 - (exp_avg.abs() / momentum_beta)).clamp(0, 1)

                #temp = (momentum_max / momentum_sum).clamp_max(1)
                #adaptivemomentum.mul_(temp)
                #momentum_sum.mul_(temp)
                exp_avg.mul_(adaptivemomentum)
                #momentum_sum.mul_(adaptivemomentum)
        return loss
'''

'''
@OPTIMIZERS.register_module()
class AdaptiveMomentumOptimizer(Optimizer):
    def __init__(self, params, lr=1e-4, weight_decay=5e-3, momentum_beta=10, max_norm=0):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum_beta=momentum_beta, momentum=0)
        self.max_norm = 0
        super(AdaptiveMomentumOptimizer, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, loss):
        
        exp_avgs = []
        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            momentum_beta = group['momentum_beta']
            for p in group['params']:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                state = self.state[p]
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                #exp_avg = state['exp_avg']
                grad = p.grad  # + weight_decay * p
                exp_avgs.append(grad)
                #exp_avgs.append((1 - (exp_avg.abs() / momentum_beta)).clamp(0, 1) * exp_avg + grad)
                # State initialization
        # total_norm = torch.max(torch.stack([p.abs().max() for p in exp_avgs]), 0)[0].item()
        total_norm = torch.norm(torch.stack([torch.norm(p, 2) for p in exp_avgs]), 2).item()
        print(loss)
        # total_norm = max(p.abs().max() for p in exp_avgs)


        loss = loss.item()
        if self.max_norm == 0:
            self.max_norm = loss
            update = True
        elif self.max_norm + 0.05 > self.max_norm * 0.9 + loss * 0.1:
            self.max_norm = self.max_norm * 0.9 + loss * 0.1
            # print(total_norm, self.max_norm)
            update = True
        else:
            print_log(f'异常loss: {loss}')
            loss = 0
            update = False
        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            momentum_beta = group['momentum_beta']
            group['momentum'] = loss
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                exp_avg = state['exp_avg']
                grad = p.grad
                if update:
                    state['step'] += 1
                    adaptivemomentum = (1 - (exp_avg.abs() / momentum_beta)).clamp(0, 1)
                    exp_avg.mul_(adaptivemomentum)
                    exp_avg.add_(grad)
                    p.mul_(1 - lr * weight_decay)
                    p.add_(exp_avg, alpha=-lr)
                else:
                    exp_avg.mul_(-1)
                    p.mul_(1 - lr * weight_decay)
                    p.add_(exp_avg, alpha=-lr)
                    exp_avg.mul_(0)
        return None
'''

@OPTIMIZERS.register_module()
class AdaptiveMomentumOptimizer(Optimizer):
    def __init__(self, params, lr=1e-4, weight_decay=5e-3, momentum_beta=10):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum_beta=momentum_beta, momentum=0)
        super(AdaptiveMomentumOptimizer, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, loss):


        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            momentum_beta = group['momentum_beta']
            group['momentum'] = loss
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                exp_avg = state['exp_avg']
                grad = p.grad
                state['step'] += 1
                adaptivemomentum = (1 - (exp_avg.abs() / momentum_beta)).clamp(0, 1)
                exp_avg.mul_(adaptivemomentum)
                exp_avg.add_(grad)
                p.mul_(1 - lr * weight_decay)
                p.add_(exp_avg, alpha=-lr)
        return None