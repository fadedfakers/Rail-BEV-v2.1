import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
import os
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.datasets.builder import DATASETS
from mmdet3d.datasets.pipelines import Compose

@DATASETS.register_module()
class RailDataset(Dataset):
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
        
        # [修复] 同样加上 flag
        self.flag = np.zeros(len(self.data_infos), dtype=np.uint8)
        
        if pipeline is not None:
            self.pipeline = Compose(pipeline)

    def load_annotations(self, ann_file):
        with open(ann_file, 'rb') as f:
            data = pickle.load(f)
        return data

    def get_ann_info(self, index):
        return dict(
            gt_bboxes_3d=None,
            gt_labels_3d=None,
            gt_poly_3d=None
        )

    def prepare_train_data(self, index):
        info = self.data_infos[index]
        input_dict = dict(
            sample_idx=info['sample_idx'],
            pts_filename=info['lidar_path'],
            img_prefix=None,
            lidar_path=info['lidar_path'],
            sweeps=[],
            timestamp=0
        )
        
        if not self.test_mode:
            ann_info = self.get_ann_info(index)
            input_dict.update(ann_info)

        if self.pipeline is None:
            return input_dict
            
        example = self.pipeline(input_dict)
        
        if self.filter_empty_gt and example is None:
            return None
            
        return example

    def prepare_test_data(self, index):
        info = self.data_infos[index]
        input_dict = dict(
            sample_idx=info['sample_idx'],
            pts_filename=info['lidar_path'],
            img_prefix=None,
            lidar_path=info['lidar_path'],
            sweeps=[],
            timestamp=0
        )
        example = self.pipeline(input_dict)
        return example

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        else:
            while True:
                data = self.prepare_train_data(idx)
                if data is None:
                    idx = self._rand_another(idx)
                    continue
                return data

    def _rand_another(self, idx):
        return np.random.randint(0, len(self.data_infos))