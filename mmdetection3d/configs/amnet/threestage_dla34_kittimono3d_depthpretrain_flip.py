_base_ = [
    '../_base_/datasets/kitti-mono3d.py',
    '../_base_/default_runtime.py', '../_base_/schedules/mmdet_schedule_1x.py'
]
work_dir = '/usr/jys/mmdetection3d/work_dirs/threestage_dla34_kittimono3d_depthpretrain/'
# resume_from2 = '/home/jys/mmdetection3d/work_dirs/threestage_dla34_kittimono3d_533/best_img_bbox/Moderate@0.7@Car@R40@AP3D_epoch_99.pth'
resume_from2 = None
gpu_ids = [0]
# fp16 = dict(loss_scale=dict(init_scale=2. ** 9, growth_factor=2.0, backoff_factor=0.5, growth_interval=500, ))
# cumulative_gradient = dict(cumulative_iters=10)
find_unused_parameters = True
custom_hooks = [dict(type='EpochFuseHook', first_val=False, priority='VERY_LOW'),
                dict(type='MyLinearMomentumEMAHook', resume_from=resume_from2, warm_up=1, momentum=0.001, priority=49)
                ]
evaluation = dict(interval=5, dynamic_intervals=[(90, 1), ],
                  save_best="img_bbox/Moderate@0.7@Car@R40@AP3D", rule='greater'
                  )
checkpoint_config = dict(interval=1, max_keep_ckpts=1)
log_config = dict(interval=5, hooks=[dict(type='TextLoggerHook'),
                                     dict(type='TensorboardLoggerHook')
                                     ])
optimizer_config = dict(_delete_=True,
                        # type="GradientCumulativeOptimizerHook", cumulative_iters=16,
                        grad_clip=dict(type='my_clip_grads', max_norm=2000, norm_type=2))
# optimizer_config = dict(grad_clip=None)
optimizer = dict(_delete_=True,
                 #   type='SGD', lr=1e-3, momentum=0.9, weight_decay=1e-4,  # nesterov=True,
                 # type='AdamW', lr=5e-4, betas=(0.9, 0.999), weight_decay=0.01,
                 # type='PID_SGD', lr=1e-5, weight_decay=0.05, pid=(1, 0.1, 0), integral_separation=(1, 0),
                 # type='ZTransferFunctionOptimizer', lr=1e-4, weight_decay=0.005,
                 # numerator=(0.15, -0.09),denominator=(1, -1.3, 0.36),
                 # numerator=(0.01,), denominator=(1, -0.99),
                 type='BallOptimizer', lr=1e-2, weight_decay=5e-4, alpha=500,
                 # type='AdaptiveMomentumOptimizer', lr=1e-2, weight_decay=1e-5, momentum_beta=0.2, max_norm=1000,
                 #  type='AdaptiveMomentumOptimizerv2', lr=1e-4, weight_decay=0.005, momentum_beta=0.5,
                 # type='SignGradOptimizer', lr=1e-4, weight_decay=0.005, momentum=0.9, beta=(1, 1),
                 # type='StateSpaceOptimizer', lr=1e-4, weight_decay=0.005,
                 # A='((0, 0, 0), (-0.1, 0.9, 0), (0.1 * lr, -0.9 * lr, 1 - lr * weight_decay))',
                 # B='(1, 0.2, -0.2*lr)',
                 # C='(0, 0, 1)',
                 # x0=('0', '0', 'p'),
                 paramwise_cfg=dict(bias_lr_mult=1, norm_decay_mult=0., bias_decay_mult=0., dcn_offset_lr_mult=1,
                                    custom_keys={'backbone': dict(lr_mult=1)}), )
# optimizer = dict(_delete_=True, type='Adam', lr=1e-4, betas=(0., 0.))
# optimizer = dict(_delete_=True, type='AdamW', lr=2e-4, betas=(0.9, 0.9), weight_decay=0.01,
#               paramwise_cfg=dict(bias_lr_mult=1., norm_decay_mult=0., bias_decay_mult=0., dcn_offset_lr_mult=1,
#                                  custom_keys={'backbone': dict(lr_mult=1)}), )
lr_config = dict(_delete_=True,
                 # policy='cyclic', target_ratio=(1, 1e-4), cyclic_times=1, step_ratio_up=2 / 3,
                 policy='step', step=[500, ],
                 # policy='FlatCosineAnnealing', min_lr_ratio=0.1, start_percent=0.5,
                 warmup='linear', warmup_iters=500, warmup_ratio=0.001,
                 )
# momentum_config = dict(policy='cyclic', target_ratio=(0.85 / 0.95, 1), cyclic_times=1, step_ratio_up=0.4, )
# lr_config = dict(policy='step', warmup=None, step=[50, ])
runner = dict(type='MyEpochBasedRunner', max_epochs=100)
# optimizer = dict(_delete_=True, type='AdamW', lr=5e-4, weight_decay=1e-5, betas=(0.9, 0.9))
LOSS_WEIGHT_RPN = 1
LOSS_WEIGHT_ROI1 = 1
LOSS_WEIGHT_ROI2 = 1
MAX_NUM_PRE_IMG = 50
# LOSS_REDUCTION = 'mean'
LOSS_REDUCTION = 'sum'
CLASS_NAMES = ['Pedestrian', 'Cyclist', 'Car']
# CLASS_NAMES = ['Car', ]
FREE_LOSS = False
AW_LOSS = True
BN_MOMENTUM = 0.1
NECK_OUT_CHANNEL = 128
HEAD_RPN_FEAT_CHANNEL = 64
HEAD_2D_FEAT_CHANNEL = 64
HEAD_3D_FEAT_CHANNEL = 64
ROI_FEAT_SIZE1 = 5
ROI_FEAT_SIZE2 = 5
USE_CLS = None
# lr_config = dict(policy='step', warmup='linear', warmup_iters=500, warmup_ratio=1.0 / 100, step=[50, ])
IMG_SIZE = (384, 1280)
DOWN_STRIDE = (8,)
# BASE_DEPTH = (1200, 140) # d*hroi
BASE_DEPTH = (1.7, 0.2)  # h3d/h2d
# BASE_DEPTH = (1.1, 0.1) # 1/h2d
# BASE_DEPTH = (40, 8)  # d
BASE_DWDH = (0.5, 0.5)
BASE_DXDY = (0., 1)
BASE_DUDV = (0., 0.2)
# BASE_DIM = (((0.84, 1.76, 0.66), (1.76, 1.74, 0.60), (3.88, 1.53, 1.63)),
#             #     ((0.2, 0.2, 0.2), (0.2, 0.2, 0.2), (0.2, 0.2, 0.2)
#             ((0.33, 0.07, 0.22), (0.10, 0.06, 0.22), (0.11, 0.09,  0.06)
#              ))
BASE_DIM = (((0.88, 1.73, 0.67), (1.78, 1.70, 0.58), (3.88, 1.63, 1.53)),
            ((0.2, 0.2, 0.2), (0.2, 0.2, 0.2), (0.2, 0.2, 0.2)))
BASE_ALPHA = (0., 0.2)  # 4bin
# BASE_ALPHA = (0., 0.25)  # sincos
# BASE_ALPHA = (0., 0.2) # hsinhcos
ALPHA_TYPE = 'my4bin'
# BASE_IOU3D = (0.5, 0.1)
BASE_IOU3D = (0, 1)
MAX_DEPTH = 600
norm_cfg = dict(type='BN', requires_grad=True, momentum=BN_MOMENTUM)
model = dict(
    type='TwoStageMono3DDetector',
    # backbone=dict(type='ResNet', depth=34, norm_eval=False, norm_cfg=norm_cfg, out_indices=(0, 1, 2, 3),init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet34')),
    # backbone=dict(type='Darknet', depth=53, norm_eval=False, out_indices=(1, 2, 3, 4, 5), init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://darknet53')),
    backbone=dict(type='DLANet', depth=34, out_indices=(3, 4, 5), norm_cfg=norm_cfg,
                  init_cfg=dict(type='Pretrained',
                                checkpoint='/usr/jys/mmdetection3d/work_dirs/pretrained_model/dla34-depth_pretrained.pth')),
    # neck=dict(type='MyFPNNeck', out_channels=NECK_OUT_CHANNEL),
    # neck=dict(type='MyDLANeck2'),
    # neck=dict(type='MyDLANeck'),
    # neck=dict(type='DLANeck2', in_channels=[32, 64, 128, 256, 512], norm_cfg=norm_cfg,),
    neck=dict(type='SimpleNeck', norm_cfg=norm_cfg, upsample='deform'),
    # neck=dict(type='MLPNeck', out_channels=NECK_OUT_CHANNEL,norm_cfg=norm_cfg,
    # neck=dict(type='DLANeck', in_channels=[128, 256, 512], start_level=0, end_level=2, use_dcn='dcn',
    #          norm_cfg=norm_cfg),
    rpn_head=dict(
        type='TwoStageRPNHead',
        num_classes=len(CLASS_NAMES),
        in_channel=[NECK_OUT_CHANNEL, ],
        # in_channel=[224, 384, 512],
        strides=DOWN_STRIDE,
        feat_channel=HEAD_RPN_FEAT_CHANNEL,
        free_loss=FREE_LOSS,
        norm_cfg=norm_cfg,
        # norm_cfg=dict(type='GN', num_groups=32),
        # norm_cfg=None,
        class_balance=False,
        loss_center_heatmap=dict(type='MyGaussianFocalLoss', loss_weight=1 * LOSS_WEIGHT_RPN, alpha=2.0,
                                 reduction=LOSS_REDUCTION),
        roi_head=dict(
            type='BBox2DRoIHead',
            use_cls=None,
            aw_loss=AW_LOSS,
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                # roi_layer=dict(type='RoIAlign', aligned=False, output_size=ROI_FEAT_SIZE1, sampling_ratio=0),
                roi_layer=dict(type='RoIAlignDeterministic', aligned=False, output_size=ROI_FEAT_SIZE1,
                               sampling_ratio=0),
                # roi_layer=dict(type='RoIPool', output_size=ROI_FEAT_SIZE1),
                out_channels=NECK_OUT_CHANNEL,
                featmap_strides=DOWN_STRIDE),
            bbox_head=dict(
                type='BBox2DHead',
                in_channels=NECK_OUT_CHANNEL,
                feat_channel=HEAD_2D_FEAT_CHANNEL,
                roi_feat_size=ROI_FEAT_SIZE1,
                num_classes=len(CLASS_NAMES),
                bbox_coder=dict(
                    type='MyBBoxCoder',
                    base_dwdh=BASE_DWDH,
                    base_dxdy=BASE_DXDY,
                ),
                norm_cfg=norm_cfg,
                # norm_cfg=dict(type='GN', num_groups=32),
                # norm_cfg=None,
                loss_wh=dict(type='MyL1Loss', loss_weight=1 * LOSS_WEIGHT_ROI1, reduction=LOSS_REDUCTION),
                loss_offset=dict(type='MyL1Loss', loss_weight=1 * LOSS_WEIGHT_ROI1, reduction=LOSS_REDUCTION),
                loss_iou2d=dict(type='MyBinaryCrossEntropyLoss', loss_weight=1 * LOSS_WEIGHT_ROI1,
                                reduction=LOSS_REDUCTION),
                loss_iou=dict(type='MyIOULoss', loss_weight=1 * LOSS_WEIGHT_ROI1, GIOU=False,
                              reduction=LOSS_REDUCTION),
                pred=[
                    # 'offset',
                    # 'wh',
                    'iou',
                    # 'union',
                    'iou2d',
                ],
                train_cfg=dict(
                    # roi_num=2000,
                    # nms=dict(type='nms', iou_threshold=0.7),
                    add_gt=True,
                    min_bbox_size=8,
                    # min_score=0.
                ),
                test_cfg=dict(
                    # roi_num=2000,
                    # nms=dict(type='nms', iou_threshold=0.7),
                    min_bbox_size=8,
                    # min_score=0.
                ),
            ),
            train_cfg=dict(max_num_pre_img=MAX_NUM_PRE_IMG),
        ),
        train_cfg=dict(
            max_num_pre_img=MAX_NUM_PRE_IMG,
            add_gt=True,
            nms_pre=512,
            # roi_num=-1,
            # min_score=0.
        ),
        test_cfg=dict(
            nms_pre=512,
            # roi_num=-1,
            # min_score=0.
        ),
    ),
    roi_head=dict(
        type='BBox3DRoIHead',
        use_cls=USE_CLS,
        aw_loss=AW_LOSS,
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            # roi_layer=dict(type='RoIAlign', output_size=ROI_FEAT_SIZE2, aligned=False, sampling_ratio=0),
            roi_layer=dict(type='RoIAlignDeterministic', output_size=ROI_FEAT_SIZE2, aligned=False, sampling_ratio=0),
            # roi_layer=dict(type='RoIPool', output_size=ROI_FEAT_SIZE2),
            out_channels=NECK_OUT_CHANNEL,
            featmap_strides=DOWN_STRIDE),
        bbox_head=dict(
            type='BBox3DHead',
            in_channels=NECK_OUT_CHANNEL,
            feat_channel=HEAD_3D_FEAT_CHANNEL,
            roi_feat_size=ROI_FEAT_SIZE2,
            num_classes=len(CLASS_NAMES),
            couple='adaptive',
            # couple='decouple',
            bbox_coder=dict(
                type='MyBBoxCoder',
                # depth_type='d*hroi',
                depth_type='h3d/h2d',
                # depth_type='1/h2d',
                # depth_type='d',
                alpha_type=ALPHA_TYPE,
                base_depth=BASE_DEPTH,
                base_dims=BASE_DIM,
                base_dwdh=BASE_DWDH,
                base_dxdy=BASE_DXDY,
                base_dudv=BASE_DUDV,
                base_alpha=BASE_ALPHA,
                base_iou3d=BASE_IOU3D,
            ),
            norm_cfg=norm_cfg,
            # norm_cfg=dict(type='GN', num_groups=32),
            # norm_cfg=None,
            loss_cls=dict(type='MyGaussianFocalLoss', loss_weight=1 * LOSS_WEIGHT_ROI2, alpha=2.0,
                          reduction=LOSS_REDUCTION),
            loss_lhw=dict(type='MyL1Loss', loss_weight=1 * LOSS_WEIGHT_ROI2, reduction=LOSS_REDUCTION),
            loss_d=dict(type='MyL1Loss', loss_weight=1 * LOSS_WEIGHT_ROI2, reduction=LOSS_REDUCTION),
            loss_uv=dict(type='MyL1Loss', loss_weight=1 * LOSS_WEIGHT_ROI2, reduction=LOSS_REDUCTION),
            loss_alpha=dict(type='MyL1Loss', loss_weight=1 * LOSS_WEIGHT_ROI2, reduction=LOSS_REDUCTION),
            # loss_alpha=dict(type='MyBinaryCrossEntropyLoss', loss_weight=1 * LOSS_WEIGHT_ROI2,
            #                 reduction=LOSS_REDUCTION),
            # loss_alpha_4bin=dict(type='MyCrossEntropyLoss', loss_weight=1 * LOSS_WEIGHT_ROI2, reduction=LOSS_REDUCTION),
            loss_alpha_4bin=dict(type='MyGaussianFocalLoss', loss_weight=1 * LOSS_WEIGHT_ROI2, alpha=2.0,
                                 reduction=LOSS_REDUCTION),
            loss_iou3d=dict(type='MyBinaryCrossEntropyLoss', loss_weight=1 * LOSS_WEIGHT_ROI2,
                            reduction=LOSS_REDUCTION),
            # loss_iou3d=dict(type='MyL1Loss', loss_weight=1 * LOSS_WEIGHT_ROI2, reduction=LOSS_REDUCTION),
            loss_xyz=dict(type='MyL1Loss', loss_weight=0.2 * LOSS_WEIGHT_ROI2, reduction=LOSS_REDUCTION),
            loss_corner=dict(type='MyL1Loss', loss_weight=0.2 * LOSS_WEIGHT_ROI2, reduction=LOSS_REDUCTION),
            pred=[
                # 'bbox2d',
                # 'd', 'uv',
                # 'lhw',
                # 'sincos',
                # 'alpha',
                # 'iou2d',
                'iou3d',
                # 'xyz'
                # 'd_score',
                # 'corner_2d',
                #  'corner',
                'union_corner',
            ],
            train_cfg=dict(
                # min_iou=0.,
                # min_score=0.
            ),
            test_cfg=dict(
                nms_2d=dict(type='nms', iou_threshold=0.7, max_num=32),
                nms_3d=dict(type='3d', iou_threshold=0.7, max_num=32),
                max_d=MAX_DEPTH,
                min_bbox_size=0,
                min_score=0.,
            ),
        ),
        train_cfg=dict(max_num_pre_img=MAX_NUM_PRE_IMG),
    ),
    train_cfg=None,
    test_cfg=None,
)
# We fixed the incorrect img_norm_cfg problem in the source code.
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# img_norm_cfg = dict(mean=[0, 0, 0], std=[255, 255, 255], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFileMono3D', to_float32=True),
    dict(
        type='LoadAnnotations3D',
        with_bbox=True,
        with_label=True,
        with_attr_label=False,
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox_depth=True),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    # dict(type='Bbox3dTo2d'),
    dict(type='RandomHSV', saturation_range=(0.5, 2.0), hue_delta=18, value_range=(0.5, 2.0)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Init'),
    # dict(type='RandomResize3d', ratio_range=(1, 1),img_shape=size),
    # dict(type='ExpandOrCrop3d', size=size),
    dict(type='UnifiedIntrinsics', size=IMG_SIZE),
    dict(type='Pad', size=IMG_SIZE),
    # dict(type='Bbox8dtoXyzxyz'),
    dict(type='Img2Cam'),
    dict(type='MakeHeatMap3dTwoStage', size=IMG_SIZE, label_num=len(CLASS_NAMES), max_num_pre_img=MAX_NUM_PRE_IMG,
         down_factor=DOWN_STRIDE, train_without_far=1e10,
         kernel_size=0.25, size_distribution=(1280000,), train_without_ignore=False, train_without_outbound=True,
         train_without_small=(8, 8), center_type='3d', beta='inf', free_loss=FREE_LOSS, iou_heat=False,
         base_depth=BASE_DEPTH, base_dims=BASE_DIM, base_alpha=BASE_ALPHA, alpha_type=ALPHA_TYPE, ),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect',
         keys=['img', 'center_heatmap_pos', 'center_heatmap_neg', 'size_heatmap', 'lhw_heatmap', 'uv_heatmap',
               'corners_2d_heatmap', 'index_heatmap',
               'cls_heatmap_pos', 'cls_heatmap_neg', 'alpha_heatmap', 'd_heatmap', 'corner_heatmap',
               'size_mask', 'bbox2d_heatmap',
               'bbox3d_heatmap', 'alpha_4bin_heatmap', 'img2cam', 'cam2img', 'xy_max', 'xy_min', 'bbox2d_mask',
               'K_out', 'pad_bias', 'scale_factor'],
         meta_keys=[])
]

test_pipeline = [
    # dict(type='LoadImageFromFileMono3D', to_float32=True),
    # dict(type='LoadAnnotations3D', with_bbox=True, with_label=True, with_attr_label=False, with_bbox_3d=True,  with_label_3d=True, with_bbox_depth=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=IMG_SIZE[::-1],
        flip=True,
        transforms=[
            dict(type='LoadImageFromFileMono3D', to_float32=True),
            # dict(type='Bbox3dTo2d'),
            # dict(type='RandomResize3d'),
            # dict(type='RandomFlip'),
            dict(type='RandomFlip3D'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Init'),
            dict(type='UnifiedIntrinsics', size=IMG_SIZE),
            dict(type='Pad', size=IMG_SIZE),
            dict(type='Img2Cam'),
            # dict(type='Bbox8dtoXyzxyz'),
            # dict(type='MakeHeatMap3dTwoStage', size=IMG_SIZE, label_num=NUM_CLASS,, max_num_pre_img=MAX_NUM_PRE_IMG, down_factor=DOWN_STRIDE,  kernel_size=0.15, size_distribution=(1280000,), train_without_ignore=True,train_without_outbound=False,train_without_small=(8, 8),base_depth=BASE_DEPTH, base_dims=base_dims, ),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect',
                 keys=['img', 'img2cam', 'cam2img', 'K_out', 'xy_max', 'xy_min',
                       'pad_bias', 'scale_factor',
                       # 'center_heatmap_pos', 'center_heatmap_neg', 'size_heatmap', 'lhw_heatmap', 'uv_heatmap','index_heatmap', 'cls_heatmap_pos', 'cls_heatmap_neg', 'sincos_heatmap', 'd_heatmap','size_mask', 'bbox2d_heatmap', 'alpha_4bin_heatmap',
                       ], meta_keys=['box_type_3d', 'flip', 'filename', 'cam2img_ori', ])
        ])
]

data = dict(
    samples_per_gpu=8, workers_per_gpu=4,
    train=dict(pipeline=train_pipeline, classes=CLASS_NAMES, ),
    val=dict(pipeline=test_pipeline, classes=CLASS_NAMES, samples_per_gpu=8, gpu_ids=gpu_ids),
    test=dict(pipeline=test_pipeline, classes=CLASS_NAMES, samples_per_gpu=8, gpu_ids=gpu_ids))
