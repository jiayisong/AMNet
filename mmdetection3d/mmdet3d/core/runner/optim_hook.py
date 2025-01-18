from mmcv.runner.hooks import HOOKS, Hook
from torch.nn.utils import clip_grad
import torch
from torch._six import inf


def my_clip_grads(parameters, total_norm, max_norm):
    total_norm = float(total_norm)
    grad_square_sum = total_norm ** 2 / 4 / max_norm / max_norm
    clip_coef = 1 + grad_square_sum
    for p in parameters:
        p.grad.detach().div_(clip_coef)
    return total_norm


def origin_clip_grads(parameters, total_norm, max_norm):
    clip_coef = max_norm / (total_norm + 1e-6)
    # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
    # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
    # when the gradients do not reside in CPU memory.
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for p in parameters:
        p.grad.detach().mul_(clip_coef_clamped.to(p.grad.device))
    return total_norm


@HOOKS.register_module()
class MyOptimizerHook(Hook):

    def __init__(self, grad_clip=None):
        self.grad_clip = grad_clip

    def clip_grads(self, params):
        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            parameters = params
            if isinstance(parameters, torch.Tensor):
                parameters = [parameters]
            parameters = [p for p in parameters if p.grad is not None]
            clip_grads_function = globals()[self.grad_clip.get('type', 'origin_clip_grads')]
            max_norm = float(self.grad_clip['max_norm'])
            norm_type = float(self.grad_clip.get('norm_type', 2))
            error_if_nonfinite = self.grad_clip.get('error_if_nonfinite', False)
            if len(parameters) == 0:
                return torch.tensor(0.)
            device = parameters[0].grad.device
            if norm_type == inf:
                norms = [p.grad.detach().abs().max().to(device) for p in parameters]
                total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
            else:
                total_norm = torch.norm(
                    torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
            if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
                raise RuntimeError(
                    f'The total norm of order {norm_type} for gradients from '
                    '`parameters` is non-finite, so it cannot be clipped. To disable '
                    'this error and scale the gradients by the non-finite norm anyway, '
                    'set `error_if_nonfinite=False`')
            if max_norm != inf:
                total_norm = clip_grads_function(parameters, total_norm, max_norm)
            return total_norm

    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()
        runner.outputs['loss'].backward()
        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                         runner.outputs['num_samples'])
        runner.optimizer.step()
        # runner.optimizer.step(float(grad_norm))
