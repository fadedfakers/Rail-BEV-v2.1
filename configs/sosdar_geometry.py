_base_ = [
    './_base_/dataset.py',
    './_base_/model.py',
    './_base_/schedule.py',
    './_base_/default_runtime.py'
]

# ==========================================================
# [1] 基础变量定义 (必须放在最前面！)
# ==========================================================
dataset_type = 'SOSDaRDataset'
data_root = '/root/autodl-tmp/FOD/SOSDaR24/'
class_names = ['car', 'pedestrian', 'obstacle']  # <--- 之前缺这个
input_modality = dict(use_lidar=True, use_camera=False) # <--- 之前也缺这个

# ==========================================================
# [2] 数据处理流水线
# ==========================================================
train_pipeline = [
    dict(type='LoadSOSDaRPCD', load_dim=4, use_dim=4),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    
    dict(type='PointsRangeFilter', point_cloud_range=[-50, -50, -5, 50, 50, 3]),
    dict(type='ObjectRangeFilter', point_cloud_range=[-50, -50, -5, 50, 50, 3]),
    dict(type='PointShuffle'),
    
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    
    dict(type='Collect3D', 
         keys=['points', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_poly_3d'], 
         meta_keys=['pts_filename', 'sample_idx', 'pcd_horizontal_flip', 
                    'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d'])
]

test_pipeline = [
    dict(type='LoadSOSDaRPCD', load_dim=4, use_dim=4),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1.0, 1.0],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(type='PointsRangeFilter', point_cloud_range=[-50, -50, -5, 50, 50, 3]),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', 
                 keys=['points'],
                 meta_keys=['pts_filename', 'sample_idx', 'pcd_horizontal_flip', 
                            'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d'])
        ])
]

# ==========================================================
# [3] 数据配置 (Batch Size = 6, Workers = 8)
# ==========================================================
data = dict(
    samples_per_gpu=6,
    workers_per_gpu=8,
    persistent_workers=True,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'sosdar24_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        box_type_3d='LiDAR'),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'sosdar24_infos_train.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'sosdar24_infos_train.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR')
)

# ==========================================================
# [4] 优化器配置 (LR = 0.003)
# ==========================================================
optimizer = dict(
    type='AdamW', 
    lr=0.003, 
    weight_decay=0.01
)
# 每 100 个 Epoch 再验证（相当于永远不验），防止验证集坏数据导致崩溃
evaluation = dict(interval=100)