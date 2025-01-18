# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import time
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_detector
from tools.misc.fuse_conv_bn import fuse_module


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet benchmark a model')
    parser.add_argument('--config',
                        # default='configs/two_stage_mono/threestage_dla34_kittimono3d_baseline.py',
                       # default='configs/two_stage_mono/threestage_dla34_kittimono3d_AdamW_GradClip_Switch_Angle_LossWeight_LossSample_Head_RD_Neck.py',
                       # default='configs/two_stage_mono/threestage_dla34_kittimono3d_AdamW_GradClip_Switch_Angle_LossWeight_LossSample_Head_RD.py',
                        default='configs/two_stage_mono/threestage_dla34_kittimono3d.py',
                        help='test config file path')
    parser.add_argument('--checkpoint',
                        # default=None,
                        #  default='work_dirs/threestage_dla34_kittimono3d_AdamW_baseline/best_img_bbox/Moderate@0.7@Car@R40@AP3D_epoch_92.pth',
                        # default='work_dirs/threestage_dla34_kittimono3d_AdamW_CradClip_Switch_Angle_LossWeight_LossSample_Head_RD_Neck_nearest/best_img_bbox/Moderate@0.7@Car@R40@AP3D_epoch_90.pth',
                        # default='work_dirs/threestage_dla34_kittimono3d_AdamW_CradClip_Switch_Angle_LossWeight_LossSample_Head_RD/best_img_bbox/Moderate@0.7@Car@R40@AP3D_epoch_90.pth',
                          default='work_dirs/threestage_dla34_kittimono3d_533/best_img_bbox/Moderate@0.7@Car@R40@AP3D_epoch_99.pth',
                        help='checkpoint file')
    parser.add_argument('--samples', default=500, help='samples to benchmark')
    parser.add_argument(
        '--log-interval', default=50, help='interval of logging')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_false',
        help='Whether to fuse conv and bn, this will slightly increase'
             'the inference speed')
    parser.add_argument(
        '--fp16',
        action='store_true',
        help='Whether to fp16, this will slightly increase'
             'the inference speed')
    args = parser.parse_args()
    return args


# @profile
def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    # dataset = build_dataset(cfg.data.test)
    val_samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    # fp16_cfg = cfg.get('fp16', None)
    if args.fp16:
        wrap_fp16_model(model)
    if args.checkpoint is not None:
        load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_module(model)
    # print(model)
    model = MMDataParallel(model, device_ids=[0])

    model.eval()

    # the first several iterations may be very slow so skip them
    num_warmup = 5
    pure_inf_time = 0

    # benchmark with several samples and take the average
    for i, data in enumerate(data_loader):

        torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            model(return_loss=False, rescale=True, **data)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        if i >= num_warmup:
            pure_inf_time += elapsed
            if (i + 1) % args.log_interval == 0:
                fps = (i + 1 - num_warmup) / pure_inf_time
                print(
                    f'Done image [{i + 1:<3}/ {args.samples}], 'f'fps: {fps:.1f} img / s, inference time: {1000 / fps:.1f} ms / img')

        if (i + 1) == args.samples:
            pure_inf_time += elapsed
            fps = (i + 1 - num_warmup) / pure_inf_time
            print(f'Overall fps: {fps:.1f} img / s, inference time: {1000 / fps:.1f} ms / img')
            break


if __name__ == '__main__':
    main()
