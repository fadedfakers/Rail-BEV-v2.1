# configs/_base_/model.py

model = dict(
    type='RailFusionNet',
    
    backbone=dict(
        type='PillarEncoder',
        voxel_size=[0.16, 0.16, 4],
        point_cloud_range=[-50, -50, -5, 50, 50, 3],
        in_channels=4,
        feat_channels=[64],
        with_distance=False,
        voxel_layer=dict(
            max_num_points=32,
            point_cloud_range=[-50, -50, -5, 50, 50, 3],
            voxel_size=[0.16, 0.16, 4],
            max_voxels=(16000, 40000))
    ),

    neck=dict(
        type='TemporalFusion',
        in_channels=64,
        out_channels=128,
        frames_num=4,
        fusion_method='conv_gru'
    ),

    # [修复] 这里改为 RailCenterHead
    bbox_head=dict(
        type='RailCenterHead', 
        in_channels=128,
        tasks=[
            dict(num_class=1, class_names=['car']),
            dict(num_class=1, class_names=['pedestrian']),
            dict(num_class=1, class_names=['obstacle']),
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)
        ),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=[0.16, 0.16]
        ),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True
    ),

    rail_head=dict(
        type='PolyHead',
        in_channels=128,
        num_polys=2,
        num_control_points=5,
        loss_poly=dict(type='ChamferDistanceLoss', loss_weight=1.0) 
    ),

    train_cfg=dict(
        pts=dict(
            point_cloud_range=[-50, -50, -5, 50, 50, 3],
            grid_size=[640, 640, 40],
            voxel_size=[0.16, 0.16, 4],
            out_size_factor=8,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        )
    ),
    test_cfg=dict(
        pts=dict(
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=[0.16, 0.16],
            nms_type='rotate',
            pre_max_size=1000,
            post_max_size=83,
            nms_thr=0.2
        )
    )
)