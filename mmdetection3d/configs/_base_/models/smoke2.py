model = dict(
    type='SMOKEMono3D',
    backbone=dict(
        type='DLANet',
        depth=34,
        in_channels=3,
        norm_cfg=dict(type='BN'),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='http://dl.yf.io/dla/models/imagenet/dla34-ba72cf86.pth'
        )),
    neck=dict(
        type='DLANeck',
        in_channels=[16, 32, 64, 128, 256, 512],
        start_level=2,
        end_level=5,
        use_dcn=False,
        norm_cfg=dict(type='BN')),
    bbox_head=dict(
        type='SMOKEMono3DHead',
        norm_cfg=dict(type='BN'),
        num_classes=3,
        in_channels=64,
        dim_channel=[3, 4, 5],
        ori_channel=[6, 7],
        stacked_convs=0,
        feat_channels=64,
        use_direction_classifier=False,
        diff_rad_by_sin=False,
        pred_attrs=False,
        pred_velo=False,
        dir_offset=0,
        strides=None,
        # group_reg_dims=(8,),
        group_reg_dims=(1, 2, 3, 2, ),  # d offset lhw sincos
        group_reg_loss_weights=(0.2, 4, 1, 0.2),
        #group_reg_loss_weights=(0, 0, 0, 1),
        cls_branch=(64,),
        reg_branch=((64,), (64,), (64,), (64,),),
        num_attrs=0,
        bbox_code_size=7,
        dir_branch=(),
        attr_branch=(),
        bbox_coder=dict(
            type='SMOKECoder',
            #type='SMOKECylinderCoder',
            base_depth=(28.01, 16.32),
            #base_depth=(0, 1),
            base_offset=(0.50, 0.29),
            base_dims=(((0.84, 1.76, 0.66), (1.76, 1.74, 0.60), (3.88, 1.53, 1.63)),
                       #     ((0.2, 0.2, 0.2), (0.2, 0.2, 0.2), (0.2, 0.2, 0.2)
                       ((0.33, 0.07, 0.22), (0.10, 0.06, 0.22), (0.11, 0.09, 0.06)
                        )),
            code_size=7, down_ratio=4),
        # loss_cls=dict(type='GaussianVariFocalLoss', loss_weight=1.0),
        loss_cls=dict(type='GaussianFocalLoss', loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_attr=None,
        conv_bias=True,
        dcn_on_last_conv=False),
    train_cfg=None,
    test_cfg=dict(min_score=0., local_maximum_kernel=1, max_per_img=512,
                  nms_3d=dict(type='3d', iou_threshold=0.7, max_num=32),
                  )
)
