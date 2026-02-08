import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
import os
# [新增] 引入 DataContainer 用于解决长短不一的数据堆叠问题
from mmcv.parallel import DataContainer as DC 
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.datasets.builder import DATASETS, PIPELINES
from mmdet3d.datasets.pipelines import Compose

# ==================================================================
# [PCD 加载器] 
# ==================================================================
@PIPELINES.register_module()
class LoadSOSDaRPCD(object):
    def __init__(self, load_dim=4, use_dim=4):
        self.load_dim = load_dim
        self.use_dim = use_dim

    def __call__(self, results):
        pts_filename = results['pts_filename']
        try:
            points = self._load_sosdar_pcd(pts_filename)
        except Exception as e:
            raise ValueError(f"PCD Load Fail: {e}")
            
        if points.shape[1] < 4:
            N = points.shape[0]
            zeros = np.zeros((N, 4 - points.shape[1]), dtype=np.float32)
            points = np.hstack([points, zeros])
            
        results['points'] = points
        results['pts_shape'] = points.shape
        from mmdet3d.core.points import LiDARPoints
        results['points'] = LiDARPoints(points, points_dim=points.shape[1])
        return results

    def _load_sosdar_pcd(self, filepath):
        with open(filepath, 'rb') as f:
            header_lines = []
            num_points = 0
            while True:
                line = f.readline()
                line_str = line.decode('utf-8', errors='ignore').strip()
                if line_str.startswith('POINTS'):
                    num_points = int(line_str.split()[-1])
                if line_str.startswith('DATA'):
                    break
            
            buffer = f.read()
            if num_points == 0: return np.zeros((0, 3), dtype=np.float32)

            point_step = len(buffer) // num_points
            if len(buffer) % num_points != 0:
                 valid_size = num_points * point_step
                 buffer = buffer[:valid_size]

            raw_data = np.frombuffer(buffer, dtype=np.uint8)
            raw_data = raw_data.reshape(num_points, point_step)
            xyz_bytes = raw_data[:, :12].tobytes()
            xyz = np.frombuffer(xyz_bytes, dtype=np.float32).reshape(-1, 3)
            return xyz

# ==================================================================
# [数据集类] 
# ==================================================================
@DATASETS.register_module()
class SOSDaRDataset(Dataset):
    def __init__(self, 
                 data_root, 
                 ann_file, 
                 pipeline, 
                 classes=None, 
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False, 
                 **kwargs):
        
        super().__init__()
        self.data_root = data_root
        self.ann_file = ann_file
        self.test_mode = test_mode
        self.modality = modality
        self.box_type_3d = box_type_3d
        self.filter_empty_gt = filter_empty_gt
        
        if classes is None:
            self.CLASSES = ['car', 'pedestrian', 'obstacle']
        else:
            self.CLASSES = classes

        self.data_infos = self.load_annotations(ann_file)
        self.flag = np.zeros(len(self.data_infos), dtype=np.uint8)
        self.total_len = len(self.data_infos)
        
        if pipeline is not None:
            new_pipeline = []
            for step in pipeline:
                if step['type'] == 'LoadPointsFromFile':
                    new_pipeline.append(dict(type='LoadSOSDaRPCD', load_dim=4, use_dim=4))
                else:
                    new_pipeline.append(step)
            self.pipeline = Compose(new_pipeline)

    def load_annotations(self, ann_file):
        with open(ann_file, 'rb') as f:
            data = pickle.load(f)
        return data

    def get_ann_info(self, index):
        info = self.data_infos[index]
        annos = info['annos']
        
        # 1. 解析障碍物 BBox
        if annos['gt_bboxes_3d'] is not None and len(annos['gt_bboxes_3d']) > 0:
            gt_bboxes = LiDARInstance3DBoxes(
                torch.tensor(annos['gt_bboxes_3d']), 
                box_dim=7, 
                origin=(0.5, 0.5, 0.5)
            )
            gt_labels = annos['gt_labels_3d']
        else:
            gt_bboxes = LiDARInstance3DBoxes(
                torch.tensor([], dtype=torch.float32), 
                box_dim=7, 
                origin=(0.5, 0.5, 0.5)
            )
            gt_labels = np.array([], dtype=np.int64)

        # 2. 解析轨道真值
        # 这里返回的是 list of tensors，长度可能不一致
        raw_polys = annos.get('gt_poly_3d', [])
        if raw_polys and len(raw_polys) > 0:
            gt_poly_3d = [torch.tensor(p, dtype=torch.float32) for p in raw_polys]
        else:
            gt_poly_3d = []

        return dict(
            gt_bboxes_3d=gt_bboxes,
            gt_labels_3d=gt_labels,
            gt_poly_3d=gt_poly_3d
        )

    def prepare_train_data(self, index):
        info = self.data_infos[index]
        
        # 确保使用绝对路径，防止找不到文件
        full_path = os.path.join(self.data_root, info['lidar_path'])
        
        input_dict = dict(
            sample_idx=info['sample_idx'],
            pts_filename=full_path,
            img_prefix=None,
            lidar_path=full_path,
            sweeps=[],
            timestamp=0
        )

        input_dict['img_shape'] = (0, 0, 3)
        input_dict['ori_shape'] = (0, 0, 3)
        input_dict['pad_shape'] = (0, 0, 3)
        input_dict['scale_factor'] = 1.0
        input_dict['img_fields'] = []
        
        if not self.test_mode:
            ann_info = self.get_ann_info(index)
            input_dict.update(ann_info)
            input_dict['ann_info'] = ann_info
            
            input_dict['bbox3d_fields'] = []
            if 'gt_bboxes_3d' in ann_info:
                input_dict['bbox3d_fields'].append('gt_bboxes_3d')

        if self.pipeline is None: return input_dict
        
        try:
            example = self.pipeline(input_dict)
            if self.filter_empty_gt and example is None: 
                return None
            
            # ==========================================================
            # [核心修复] 给 gt_poly_3d 穿上防弹衣 (DataContainer)
            # stack=False: 告诉 PyTorch "别把它们堆在一起，因为长度不一样！"
            # ==========================================================
            if 'gt_poly_3d' in example:
                example['gt_poly_3d'] = DC(example['gt_poly_3d'], stack=False, cpu_only=False)
            
            return example
        except Exception as e:
            return None

    def prepare_test_data(self, index):
        info = self.data_infos[index]
        full_path = os.path.join(self.data_root, info['lidar_path'])

        input_dict = dict(
            sample_idx=info['sample_idx'],
            pts_filename=full_path,
            img_prefix=None,
            lidar_path=full_path,
            sweeps=[],
            timestamp=0,
            # 必须伪造这些字段，否则 MMCV 的 Collate 会因为找不到字段而报错
            img_shape=(0, 0, 3),
            ori_shape=(0, 0, 3),
            pad_shape=(0, 0, 3),
            scale_factor=1.0,
            img_fields=[]
        )
        try:
            return self.pipeline(input_dict)
        except Exception as e:
            # 打印一下到底是哪个文件坏了，方便以后排查，但返回 None 让外部重试
            # print(f"❌ 验证集加载失败 Index {index}: {e}")
            return None

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        # 终极死循环重试机制：只要没拿到合法数据，就永远不返回
        while True:
            data = self.prepare_test_data(idx) if self.test_mode else self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def _rand_another(self, idx):
        return np.random.randint(0, len(self.data_infos))
    def evaluate(self, results, metric='mIoU', logger=None, **kwargs):
        """评估轨道线 mIoU"""
        from mmdet3d.core.evaluation.rail_utils import calculate_rail_iou # 假设你已有或需实现此工具
        
        print("\n\n📊 开始计算轨道几何 mIoU...")
        
        ious = []
        for i, result in enumerate(results):
            # 获取预测的轨道线 (假设模型输出 result['pts_rail'])
            pred_rails = result.get('pts_rail', []) 
            
            # 获取真值
            ann_info = self.get_ann_info(i)
            gt_rails = ann_info.get('gt_poly_3d', [])
            
            if len(gt_rails) == 0:
                continue
                
            # 计算这一帧的 IoU (这里需要一套几何匹配算法)
            # 简化逻辑：将线转为 BEV 掩码或计算点到线的平均距离
            frame_iou = self._calc_geometry_iou(pred_rails, gt_rails)
            ious.append(frame_iou)
            
        miou = np.mean(ious) if ious else 0
        print(f"✅ Phase 1 几何评估完成 | mIoU: {miou:.4f}")
        
        return {'mIoU': miou}