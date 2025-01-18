from mmcv.runner import HOOKS, Hook, EvalHook, CheckpointHook
import os
import torch


@HOOKS.register_module()
class EpochFuseHook(Hook):

    def __init__(self, first_val):
        self.first_val = first_val

    def before_run(self, runner):
        if self.first_val:
            for hook in runner._hooks:
                if isinstance(hook, EvalHook):
                    hook._do_evaluate(runner)
        # self.after_run(runner)

    def after_run(self, runner):
        return
        models = []
        losses = []
        epochs = []
        for checkpoint in os.listdir(runner.work_dir):
            if 'epoch' in checkpoint:
                checkpoint = runner.load_checkpoint(os.path.join(runner.work_dir, checkpoint))
                model = checkpoint['state_dict']
                loss = checkpoint['meta']['iou3d']
                epoch = checkpoint['meta']['epoch']
                if epoch >= -1:
                    models.append(model)
                    losses.append(loss)
                    epochs.append(epoch)

        weight = self.weight_compute(epochs, losses)
        for i in sorted(range(len(epochs)), key=lambda k: epochs[k]):
            print(f'epoch:{epochs[i]} loss:{losses[i]} weight:{weight[i]}')
        with torch.no_grad():
            for name, parameters in runner.model.named_parameters():
                p = torch.zeros_like(parameters)
                for alpha, x in zip(weight, models):
                    if name in x.keys():
                        alpha_temp = x[name].to(p) * alpha
                    elif ('module.' + name) in x.keys():
                        alpha_temp = x['module.' + name].to(p) * alpha
                    elif name[7:] in x.keys():
                        alpha_temp = x[name[7:]].to(p) * alpha
                    else:
                        raise RuntimeError(f'{name} is not in checkpoint')
                    p = p + alpha_temp
                parameters.data = p
            for name, buffers in runner.model.named_buffers():
                if ('running_mean' in name) or ('running_var' in name):
                    b = torch.zeros_like(buffers)
                    for alpha, x in zip(weight, models):
                        if name in x.keys():
                            alpha_temp = x[name].to(p) * alpha
                        elif ('module.' + name) in x.keys():
                            alpha_temp = x['module.' + name].to(p) * alpha
                        elif name[7:] in x.keys():
                            alpha_temp = x[name[7:]].to(p) * alpha
                        else:
                            raise RuntimeError(f'{name} is not in checkpoint')
                        b = b + alpha_temp
                    buffers.data = b

        for hook in runner._hooks:
            if isinstance(hook, EvalHook):
                hook._do_evaluate(runner)
                runner.save_checkpoint(runner.work_dir, filename_tmpl='fuse_{}.pth', save_optimizer=False, meta=None,
                                       create_symlink=True)

    def weight_compute(self, epoch, loss):
        weight = []
        avg = sum(loss) / len(loss)
        # losses = [sum(losses) / len(losses) - i for i in losses]
        # loss_list = [(i - max(loss_list)) * (0 - 1) / (max(loss_list) - min(loss_list)) + 0 for i in loss_list]
        for e, l in zip(epoch, loss):
            if e >= 0 and l >= avg:
                weight.append(l-avg)
            else:
                weight.append(0)
        weight = [i / sum(weight) for i in weight]
        return weight

    def before_epoch(self, runner):
        pass

    def after_epoch(self, runner):
        pass

    def before_iter(self, runner):
        pass

    def after_iter(self, runner):
        pass
