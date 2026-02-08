import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule, auto_fp16
from mmdet3d.models.builder import NECKS

# [核心修复] 必须注册到 NECKS
@NECKS.register_module()
class TemporalFusion(BaseModule):
    def __init__(self,
                 in_channels=64,
                 out_channels=128,
                 frames_num=4,
                 fusion_method='conv_gru', # 'conv_gru', 'cat', 'max'
                 init_cfg=None):
        super(TemporalFusion, self).__init__(init_cfg)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.frames_num = frames_num
        self.fusion_method = fusion_method
        
        if fusion_method == 'conv_gru':
            self.fusion_layer = ConvGRU(in_channels, out_channels)
        elif fusion_method == 'cat':
            # 简单的拼接融合
            self.fusion_layer = nn.Sequential(
                nn.Conv2d(in_channels * frames_num, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            # 默认 1x1 卷积升维
            self.fusion_layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

    @auto_fp16()
    def forward(self, x):
        """
        x: [B, C, H, W]  (这里的 B 包含了 batch_size * frames_num)
        """
        # 假设输入 x 是堆叠的帧
        # 我们需要将其 reshape 回 [Batch, Frames, C, H, W]
        # 但要注意：Backbone 输出的是 [B*T, C, H, W]
        
        # 简单起见，如果 batch 里的帧是混合的，且我们知道 T=4
        # 这里的 reshape 需要非常小心，取决于 Dataset 是如何堆叠的
        # SOSDaRDataset 目前输出的是单帧，如果是多帧训练，数据加载器会堆叠
        
        # 暂时简化逻辑：假设输入就是单帧特征，不做复杂时序融合，直接升维
        # 如果确实要跑时序，需要 Dataset 提供 [B, T, ...] 的数据
        
        if self.fusion_method == 'conv_gru':
            # 这里的 ConvGRU 需要 T 维度
            # 临时 Mock：把 x 当作当前帧，hidden state 设为 0
            # 真正的时序融合需要重写 Dataset
            return self.fusion_layer(x) 
            
        elif self.fusion_method == 'cat':
             return self.fusion_layer(x)
             
        else:
             return self.fusion_layer(x)

class ConvGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super(ConvGRU, self).__init__()
        self.padding = kernel_size // 2
        self.hidden_dim = hidden_dim
        
        # Reset Gate
        self.conv_r = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=self.padding)
        # Update Gate
        self.conv_z = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=self.padding)
        # New State
        self.conv_h = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=self.padding)

    def forward(self, x, h=None):
        if h is None:
            h = torch.zeros(x.size(0), self.hidden_dim, x.size(2), x.size(3), device=x.device)
            
        # 简单的单步 GRU 实现 (用于特征升维)
        combined = torch.cat([x, h], dim=1)
        r = torch.sigmoid(self.conv_r(combined))
        z = torch.sigmoid(self.conv_z(combined))
        
        combined_new = torch.cat([x, r * h], dim=1)
        h_tilde = torch.tanh(self.conv_h(combined_new))
        
        h_next = (1 - z) * h + z * h_tilde
        return h_next