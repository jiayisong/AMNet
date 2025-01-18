# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import platform
import shutil
import random
import numpy as np
import torch

import mmcv
from mmcv.runner.epoch_based_runner import EpochBasedRunner
from mmcv.runner.builder import RUNNERS
from mmcv.runner.checkpoint import save_checkpoint
from mmcv.runner.utils import get_host_info
from mmcv.runner.hooks import HOOKS, Hook

@RUNNERS.register_module()
class MyEpochBasedRunner(EpochBasedRunner):

    def run_iter(self, data_batch, train_mode, **kwargs):
        super(MyEpochBasedRunner, self).run_iter(data_batch, train_mode, **kwargs)
        '''
        max_grad = 0
        max_grad_name = ''
        for name, p in self.model.named_parameters():
            if p.grad is not None:
                gm = p.grad.detach().norm()
                if gm > max_grad:
                    max_grad_name = name
                    max_grad = gm
        print(max_grad_name, max_grad)
        '''
    def register_optimizer_hook(self, optimizer_config):
        if optimizer_config is None:
            return
        if isinstance(optimizer_config, dict):
            optimizer_config.setdefault('type', 'MyOptimizerHook')
            hook = mmcv.build_from_cfg(optimizer_config, HOOKS)
        else:
            hook = optimizer_config
        self.register_hook(hook, priority='ABOVE_NORMAL')

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='epoch_{}.pth',
                        save_optimizer=True,
                        meta=None,
                        create_symlink=True):
        """Save the checkpoint.

        Args:
            out_dir (str): The directory that checkpoints are saved.
            filename_tmpl (str, optional): The checkpoint filename template,
                which contains a placeholder for the epoch number.
                Defaults to 'epoch_{}.pth'.
            save_optimizer (bool, optional): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            meta (dict, optional): The meta information to be saved in the
                checkpoint. Defaults to None.
            create_symlink (bool, optional): Whether to create a symlink
                "latest.pth" to point to the latest checkpoint.
                Defaults to True.
        """
        if meta is None:
            meta = {}
        elif not isinstance(meta, dict):
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')
        if self.meta is not None:
            meta.update(self.meta)
            # Note: meta.update(self.meta) should be done before
            # meta.update(epoch=self.epoch + 1, iter=self.iter) otherwise
            # there will be problems with resumed checkpoints.
            # More details in https://github.com/open-mmlab/mmcv/pull/1108
        meta.update(epoch=self.epoch + 1, iter=self.iter)
        meta.update(
            random_state=(random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state()))
        if create_symlink:
            self.log_buffer.average(0)
            for k, v in self.log_buffer.output.items():
                meta[k] = v
            self.log_buffer.clear_output()
        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            dst_file = osp.join(out_dir, 'latest.pth')
            if platform.system() != 'Windows':
                mmcv.symlink(filename, dst_file)
            else:
                shutil.copy(filepath, dst_file)

    def resume(self,
               checkpoint,
               resume_optimizer=True,
               map_location='cpu'):
        super(MyEpochBasedRunner, self).resume(checkpoint, resume_optimizer, map_location)
        random_state, np_random_state, torch_random_state, torch_cuda_random_state = self.meta.pop('random_state')
        torch_random_state = torch_random_state.cpu()
        torch_cuda_random_state = torch_cuda_random_state.cpu()
        random.setstate(random_state)
        np.random.set_state(np_random_state)
        torch.set_rng_state(torch_random_state)
        torch.cuda.set_rng_state(torch_cuda_random_state)

    def load_checkpoint(self,
                        filename,
                        map_location='cpu',
                        strict=False,
                        revise_keys=[(r'^module\.', '')]):

        checkpoint = super(MyEpochBasedRunner, self).load_checkpoint(filename,
                                                                     map_location=map_location,
                                                                     strict=strict,
                                                                     revise_keys=revise_keys)
        return checkpoint
