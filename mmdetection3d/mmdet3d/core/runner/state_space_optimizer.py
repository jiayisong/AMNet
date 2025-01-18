import math
from mmcv.runner.optimizer.builder import OPTIMIZERS
import torch
from torch.optim import _functional as F
# from torch.optim.adamw import AdamW
from torch.optim.optimizer import Optimizer
from torch.optim.sgd import SGD


@OPTIMIZERS.register_module()
class StateSpaceOptimizer(Optimizer):
    def __init__(self, params, lr=1e-4, weight_decay=5e-3,
                 A='((0.9, 0), (-0.9 * lr, 1 - lr * weight_decay))',
                 B='(0.1, -0.1*lr)',
                 C='(0, 1)',
                 x0=('0', 'p'),
                 # A='lambda lr, weight_decay: ((0.9, 0, 0),\
                 #                             (-0.9 * lr, 1 - lr * weight_decay, 0),\
                 #                             (-0.9 * lr * 0.001, (1 - lr * weight_decay) * 0.001, 0.999))',
                 # B='lambda lr, weight_decay: (1, -lr, -lr * 0.001)',
                 # C='lambda lr, weight_decay: (0, 0, 1)',
                 # x0=('0', 'p', 'p'),
                 ):
        exec(f'self.A = lambda lr, weight_decay: {A}')
        exec(f'self.B = lambda lr, weight_decay: {B}')
        exec(f'self.C = lambda lr, weight_decay: {C}')
        self.x0 = x0
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(StateSpaceOptimizer, self).__init__(params, defaults)

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
            A = self.A(lr, weight_decay)
            B = self.B(lr, weight_decay)
            C = self.C(lr, weight_decay)
            n = len(A)
            for p in group['params']:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state['state'] = self.init_x(p, p.grad)
                u = p.grad
                x = state['state']
                x = self.update_x(A, x, B, u)
                # self.update_x(A, x, B, u)
                y = self.update_y(C, x)
                self.state[p]['state'] = x
                p.data = y.data
        return loss

    def init_x(self, p, grad):
        x = []
        for i in self.x0:
            if i == '0':
                x.append(torch.zeros_like(p, memory_format=torch.preserve_format))
            elif i == 'p':
                x.append(p.clone().detach())
            elif i == 'g':
                x.append(grad.clone().detach())
        return x

    def update_x(self, A, x, B, u):
        x_new = []
        for a, b in zip(A, B):
            x_i_new = b * u
            for a_i, x_i in zip(a, x):
                x_i_new = x_i_new + a_i * x_i
            x_new.append(x_i_new)
        # for i, x_new_i in enumerate(x_new):
        #    x[i].data = x_new_i
        return x_new

    def update_y(self, C, x):
        y = 0
        for c_i, x_i in zip(C, x):
            y = y + c_i * x_i
        return y
