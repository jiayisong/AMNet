import math
from mmcv.runner.optimizer.builder import OPTIMIZERS
import torch
from torch.optim import _functional as F
from torch.optim.adamw import AdamW
from torch.optim.optimizer import Optimizer
from torch.optim.sgd import SGD

'''
@OPTIMIZERS.register_module()
class BallOptimizer(Optimizer):
    def __init__(self, params, lr=1e-4, weight_decay=5e-3, mu=0.9, g=9.8, alpha=1):
        defaults = dict(lr=lr, weight_decay=weight_decay, mu=mu, g=g, alpha=alpha)
        super(BallOptimizer, self).__init__(params, defaults)

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
        parameters = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    parameters.append(p)
        if len(parameters) == 0:
            return
        device = parameters[0].grad.device

        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2).to(device) for p in parameters]),
                                2).square()

        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            alpha = group['alpha']
            mu = group['mu']
            g = group['g']
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
                total_norm_sqrt = (1 + total_norm).sqrt()
                t = total_norm_sqrt * math.sqrt(lr)
                # print(t)
                grad = p.grad
                v = state['v']
                # x = state['x']
                a_g = -g * grad / (1 + total_norm)
                a_f1 = mu * g * grad / (total_norm_sqrt * (total_norm + total_norm.square()).sqrt())
                a = a_g + a_f1
                # print((alpha * t * v.abs()).clamp_max(1).min())
                # print(t * alpha)
                v.mul_(1 - (alpha * t * v.abs()).clamp_max(1)).add_(a, alpha=t)
                p.mul_(1 - (t * weight_decay).clamp_max(1)).add_(v, alpha=t)
        return loss
'''

'''
@OPTIMIZERS.register_module()
class BallOptimizer(Optimizer):
    def __init__(self, params, lr=1e-4, weight_decay=5e-3, mu=0.9, g=9.8, alpha=1, beta=1):
        defaults = dict(lr=lr, weight_decay=weight_decay, mu=mu, g=g, alpha=alpha, beta=beta)
        super(BallOptimizer, self).__init__(params, defaults)

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
            mu = group['mu']
            g = group['g']
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
                t = lr
                # print(t)
                grad = p.grad * beta
                grad_square_sum = total_norm * beta * beta
                grad_square_sum_sqrt = grad_norm * beta
                v = state['v']
                # x = state['x']
                a_g = -g * grad / (1 + grad_square_sum)
                #a_f1 = mu * g * grad / (grad_square_sum_sqrt * (grad_square_sum + grad_square_sum ** 2) ** 0.5)
                a = a_g# + a_f1
                # print((alpha * t * v.abs()).clamp_max(1).min())
                # print(t * alpha)
                v.mul_(1 - (alpha * t * v.abs()).clamp_max(1)).add_(a, alpha=t)
                p.mul_(1 - (t * weight_decay)).add_(v, alpha=t)
        return None
'''


@OPTIMIZERS.register_module()
class BallOptimizer(Optimizer):
    def __init__(self, params, lr=1e-4, weight_decay=5e-3,alpha=1):
        defaults = dict(lr=lr, weight_decay=weight_decay, alpha=alpha)
        super(BallOptimizer, self).__init__(params, defaults)

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
        # if 'v' not in self.state:
        #    self.state['v'] = 0
        # total_norm = grad_norm ** 2
        # dot = self.compute_dot()
        # v_norm = self.compute_v_norm()
        # self.transform_v(total_norm, dot)
        # v_norm_2 = self.compute_v_norm()
        # zoom_a = v_norm / max(v_norm_2, 1e-8)
        # self.zoom_v(zoom_a)
        # print((1 - (dot / (max(v_norm * (total_norm*0.05 * 0.05+1)**0.5, 1e-8))) ** 2) ** 0.5, v_norm_2 / (max(v_norm, 1e-8)))
        # print('n',(total_norm*0.05*0.05+1)**0.5)
        # print('n^2', (total_norm * 0.05 * 0.05 + 1))
        # print('v1v2', v_norm**2, v_norm_2**2)
        # print(dot, self.compute_dot())
        # print(90-math.acos(dot/(v_norm*grad_norm*0.05+ 1e-8))*180/math.pi, v_norm_2 / (v_norm + 1e-8), self.compute_dot())
        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            alpha = group['alpha']
            #mu = group['mu']
            #g_max = group['g_max']
            #beta = group['beta']
            # group['momentum'] = zoom_a
            for p in group['params']:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                state = self.state[p]
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state['v'] = torch.zeros_like(p)
                t = lr
                # t = 1e-2
                #grad = p.grad * beta  # + p * weight_decay
                # print(p.grad.abs().mean()/p.abs().mean())
                # if g_max > 0:
                #     grad_square_sum = total_norm * beta * beta / 4 / g_max / g_max
                #     a_g = - grad / (1 + grad_square_sum)  # ** 0.5
                # else:
                #     a_g = - grad
                a_g = - p.grad
                # grad_square_sum_sqrt = grad_norm * beta
                v = state['v']

                # a_g = -g * grad / 2 * math.exp(1 - beta * grad_norm)
                # a_g = -g * grad * (1 - beta * grad_norm / 2)
                # a_g = -g * grad
                # a_g = -0.5 * grad * max(0, 1 - grad_norm * 0.0002)
                # a_g = -g * grad / (1 + grad**2)
                # a_f1 = mu * g * grad / (grad_square_sum_sqrt * (grad_square_sum + grad_square_sum ** 2) ** 0.5)
                a = a_g  # + a_f1
                gamma = 1 - (alpha * t * v.abs()).clamp_max(1)
                v.mul_(gamma).add_(a, alpha=t)
                p.mul_(1 - weight_decay * lr).add_(v, alpha=lr)
                group['momentum'] = float(gamma.mean())
                # p.add_(v, alpha=t)
                # print(p.abs().mean())
        # self.state['v'] *= (1 - min((alpha * t * abs(self.state['v'])), 1))
        # self.state['v'] += (-g * t * grad_square_sum / (1 + grad_square_sum))
        return loss

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
