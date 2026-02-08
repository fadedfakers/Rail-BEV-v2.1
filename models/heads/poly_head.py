import torch
import torch.nn as nn
from mmdet3d.models.builder import HEADS, build_loss
from mmcv.runner import BaseModule

class ChamferDistanceLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, source, target):
        """
        计算两组点云的 Chamfer Distance
        source: [B, N, 3] (Pred)
        target: [B, M, 3] (GT)
        """
        # 简单的 L2 Chamfer 实现
        dists1 = torch.cdist(source, target) # [B, N, M]
        min_dists1, _ = torch.min(dists1, dim=2) # [B, N]
        term1 = torch.mean(min_dists1, dim=1) # [B]

        min_dists2, _ = torch.min(dists1, dim=1) # [B, M]
        term2 = torch.mean(min_dists2, dim=1) # [B]

        return self.loss_weight * (torch.mean(term1) + torch.mean(term2))

@HEADS.register_module()
class PolyHead(BaseModule):  # <--- 注意：类名是 PolyHead
    def __init__(self, 
                 in_channels=128, 
                 num_polys=2,     # 左右两根轨道
                 num_control_points=5, # 每根轨道5个控制点
                 hidden_dim=256,
                 loss_poly=dict(type='ChamferDistanceLoss', loss_weight=1.0),
                 init_cfg=None):
        super(PolyHead, self).__init__(init_cfg)
        
        self.num_polys = num_polys
        self.num_points = num_control_points
        self.out_dim = num_polys * num_control_points * 3 # 输出 (x,y,z)
        
        # 1. 几何特征提取 (Global Context)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 2. MLP 回归器
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.out_dim)
        )
        
        # 3. 损失函数
        if loss_poly['type'] == 'ChamferDistanceLoss':
            self.loss_func = ChamferDistanceLoss(loss_weight=loss_poly['loss_weight'])
        else:
            self.loss_func = nn.MSELoss()

    def forward(self, x):
        """
        Input: [B, C, H, W]
        Output: [B, num_polys, num_points, 3]
        """
        B = x.shape[0]
        feat = self.global_pool(x).view(B, -1)
        points_pred = self.mlp(feat)
        points_pred = points_pred.view(B, self.num_polys, self.num_points, 3)
        return points_pred

    def loss(self, points_pred, gt_poly_3d):
        loss_dict = dict()
        total_loss = 0
        
        # 假设 gt_poly_3d 是一个列表，每个元素是 tensor [M, 3]
        for i in range(len(points_pred)):
            pred = points_pred[i].view(-1, 3) # [10, 3] 合并左右轨道的点
            
            target = gt_poly_3d[i] 
            
            # 处理 list 类型的 gt (可能包含多段 poly)
            if isinstance(target, list):
                if len(target) > 0:
                    if torch.is_tensor(target[0]):
                        target = torch.cat(target, dim=0)
                    else:
                        # 如果完全是空的或者是 numpy
                        target = torch.tensor(target, device=pred.device)
                else:
                    # 无真值时跳过或给0 loss
                    continue
            
            if not torch.is_tensor(target):
                 target = torch.tensor(target, device=pred.device)

            loss = self.loss_func(pred.unsqueeze(0), target.unsqueeze(0))
            total_loss += loss
            
        loss_dict['loss_poly'] = total_loss / (len(points_pred) + 1e-6)
        return loss_dict