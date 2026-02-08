import torch
import torch.nn as nn
from mmdet3d.models.builder import HEADS, build_loss
from mmcv.runner import BaseModule

# [关键修改] 类名必须改为 RailCenterHead，否则会和 mmdet3d 内置的冲突
@HEADS.register_module()
class RailCenterHead(BaseModule):
    def __init__(self, 
                 in_channels=128, 
                 tasks=None, 
                 common_heads=dict(), 
                 share_conv_channel=64, 
                 bbox_coder=None, 
                 loss_cls=dict(type='GaussianFocalLoss', reduction='mean'), 
                 loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25), 
                 norm_bbox=True, 
                 init_cfg=None):
        super(RailCenterHead, self).__init__(init_cfg)
        
        self.in_channels = in_channels
        self.tasks = tasks
        self.common_heads = common_heads
        self.norm_bbox = norm_bbox
        self.bbox_coder = bbox_coder
        
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, share_conv_channel, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(share_conv_channel),
            nn.ReLU(inplace=True)
        )
        
        self.task_heads = nn.ModuleList()
        for task in tasks:
            heads = nn.ModuleDict()
            heads['hm'] = nn.Sequential(
                nn.Conv2d(share_conv_channel, share_conv_channel, 3, padding=1),
                nn.BatchNorm2d(share_conv_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(share_conv_channel, task['num_class'], 1)
            )
            for head_name, head_channels in common_heads.items():
                heads[head_name] = nn.Sequential(
                    nn.Conv2d(share_conv_channel, share_conv_channel, 3, padding=1),
                    nn.BatchNorm2d(share_conv_channel),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(share_conv_channel, head_channels[-1], 1)
                )
            self.task_heads.append(heads)

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)

    def forward(self, x):
        ret_dicts = []
        x = self.shared_conv(x)
        for task_head in self.task_heads:
            task_dict = {}
            for head_name, head_layer in task_head.items():
                task_dict[head_name] = head_layer(x)
                if head_name == 'hm':
                    task_dict[head_name] = torch.sigmoid(task_dict[head_name])
                    task_dict[head_name] = torch.clamp(task_dict[head_name], min=1e-4, max=1-1e-4)
            ret_dicts.append(task_dict)
        return ret_dicts

    def loss(self, preds_dicts, gt_bboxes_3d, gt_labels_3d):
        loss_dict = dict()
        # 占位 Loss 以防报错，实际需要完善 Target Assignment
        device = preds_dicts[0]['hm'].device
        loss_dict['loss_heatmap'] = torch.tensor(0.0, device=device, requires_grad=True)
        loss_dict['loss_bbox'] = torch.tensor(0.0, device=device, requires_grad=True)
        return loss_dict