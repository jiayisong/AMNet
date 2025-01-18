_base_ = [
    '../_base_/datasets/kitti-mono3d.py', '../_base_/models/smoke2.py',
    '../_base_/default_runtime.py'
]
work_dir = '/mnt/jys/mmdetection3d/work_dirs/smoke_119/'
# resume_from2 = '/mnt/jys/mmdetection3d/work_dirs/smoke_86/best_img_bbox/Moderate@0.7@Car@R40@AP3D_epoch_30.pth'
resume_from2 = None
gpu_ids = [0]
BATCH_SIZE = 8
# fp16 = dict(loss_scale=dict(init_scale=2. ** 9, growth_factor=2.0, backoff_factor=0.5, growth_interval=500, ))
# cumulative_gradient = dict(cumulative_iters=10)
find_unused_parameters = True
custom_hooks = [dict(type='EpochFuseHook', first_val=False, priority='VERY_LOW'),
                dict(type='MyLinearMomentumEMAHook', resume_from=resume_from2, warm_up=1, momentum=0.001, priority=49)
                ]
evaluation = dict(interval=5 * BATCH_SIZE // 8, dynamic_intervals=[(90 * BATCH_SIZE // 8, BATCH_SIZE // 8), ],
                  save_best="img_bbox/Moderate@0.7@Car@R40@AP3D", rule='greater'
                  )
checkpoint_config = dict(interval=1, max_keep_ckpts=1)
log_config = dict(interval=5, hooks=[dict(type='TextLoggerHook'),
                                     dict(type='TensorboardLoggerHook')
                                     ])
optimizer_config = dict(grad_clip=dict(type='origin_clip_grads', max_norm=2000, norm_type=2))
optimizer = dict(type='AdamW', lr=2.5e-4, betas=(0.9, 0.999), weight_decay=0.02,
                 paramwise_cfg=dict(bias_lr_mult=1, norm_decay_mult=0., bias_decay_mult=0., dcn_offset_lr_mult=1,
                                    custom_keys={'backbone': dict(lr_mult=1)}), )
# optimizer = dict(_delete_=True, type='Adam', lr=1e-4, betas=(0., 0.))
# optimizer = dict(_delete_=True, type='AdamW', lr=2e-4, betas=(0.9, 0.9), weight_decay=0.01,
#               paramwise_cfg=dict(bias_lr_mult=1., norm_decay_mult=0., bias_decay_mult=0., dcn_offset_lr_mult=1,
#                                  custom_keys={'backbone': dict(lr_mult=1)}), )
lr_config = dict(
    # policy='cyclic', target_ratio=(1, 1e-4), cyclic_times=1, step_ratio_up=2 / 3,
    policy='step', step=[90 * BATCH_SIZE // 8, ],
    # policy='FlatCosineAnnealing', min_lr_ratio=0.1, start_percent=0.5,
    warmup='linear', warmup_iters=500, warmup_ratio=0.001,
)
# momentum_config = dict(policy='cyclic', target_ratio=(0.85 / 0.95, 1), cyclic_times=1, step_ratio_up=0.4, )
# lr_config = dict(policy='step', warmup=None, step=[50, ])
runner = dict(type='MyEpochBasedRunner', max_epochs=100 * BATCH_SIZE // 8)
# optimizer
TRAIN_IMG_SIZE = (384, 1280)
TEST_IMG_SIZE = (384, 1280)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
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
    dict(type='RandomHSV', saturation_range=(0.5, 2.0), hue_delta=18, value_range=(0.5, 2.0)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='K_out'),
    dict(type='CylinderAndImgToTensor', size=TRAIN_IMG_SIZE,
         intrinsics=((881.8, 0.0, 639.5), (0.0, 801.7, 172.854), (0.0, 0.0, 1.0)), random_shift=(0, 0),
         random_scale=(1, 1), cycle=False),
    # dict(type='UnifiedIntrinsics', size=IMG_SIZE,
    #      intrinsics=((721.5377, 0.0, 639.5), (0.0, 721.5377, 174.854), (0.0, 0.0, 1.0)), random_shift=(4, 0),
    #      random_scale=(1, 1)),
    # dict(type='Pad', size=IMG_SIZE),
    dict(type='Img2Cam'),
    dict(type='CatDepth', img_size=TRAIN_IMG_SIZE, d_max=50, cv=172.854),
    dict(type='SMOKEGetTarget', down_ratio=4, num_classes=3, train_without_small=(8, 8), train_without_far=800,
         train_without_centerout='img', max_objs=50),
    # dict(type='ImageToTensor', keys=['img']),
    dict(
        type='Collect3D',
        keys=[
            'img', 'center_heatmap_target', 'gt_centers2d', 'gt_labels3d', 'indices', 'reg_indices',
            'gt_locs', 'gt_dims', 'gt_yaws', 'gt_cors', 'gt_offsets', 'gt_depths', 'img2cam', 'cam2img', 'K_out',
            'z_indice', 'normal_uv', 'gt_h2d'
        ],
        meta_keys=('box_type_3d',),
    ),
]
test_pipeline = [
    dict(type='LoadImageFromFileMono3D', to_float32=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=TEST_IMG_SIZE[::-1],
        flip=False,
        transforms=[
            dict(type='K_out'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='CylinderAndImgToTensor', size=TEST_IMG_SIZE,
                 intrinsics=((881.8, 0.0, 639.5), (0.0, 801.7, 172.854), (0.0, 0.0, 1.0))),
            # dict(type='UnifiedIntrinsics', size=IMG_SIZE,
            #      intrinsics=((721.5377, 0.0, 639.5), (0.0, 721.5377, 172.854), (0.0, 0.0, 1.0)), random_shift=(0, 0),
            #      random_scale=(1, 1)),
            # dict(type='Pad', size=IMG_SIZE),
            dict(type='Img2Cam'),
            # dict(type='ImageToTensor', keys=['img']),
            dict(type='CatDepth', img_size=TEST_IMG_SIZE, d_max=50, cv=172.854),
            dict(type='SMOKEGetTarget', down_ratio=4, test_mode=True),
            dict(type='Collect3D', keys=['img', 'cam2img', 'img2cam', 'K_out',
                                         'normal_uv', ], meta_keys=('box_type_3d',), ),
        ])
]
data = dict(
    samples_per_gpu=BATCH_SIZE,
    workers_per_gpu=BATCH_SIZE // 2,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline, samples_per_gpu=8, gpu_ids=gpu_ids),
    test=dict(pipeline=test_pipeline, samples_per_gpu=8, gpu_ids=gpu_ids))

model = dict(
    backbone=dict(in_channels=4, init_cfg=dict(type='Pretrained',
                                               checkpoint='/home/jys/mmdetection3d/work_dirs/pretrained_model/dla34-ba72cf86-base_layer_channel-4.pth')),
    bbox_head=dict(bbox_coder=dict(type='SMOKECylinderCoder', ))
)
