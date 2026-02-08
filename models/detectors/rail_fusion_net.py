import torch
from mmdet3d.models.detectors import Base3DDetector
from mmdet3d.models import builder
from mmdet3d.models.builder import DETECTORS

@DETECTORS.register_module()
class RailFusionNet(Base3DDetector):
    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 rail_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 pretrained=None):
        super(RailFusionNet, self).__init__(init_cfg)
        
        self.backbone = builder.build_backbone(backbone)
        self.neck = builder.build_neck(neck)
        self.bbox_head = builder.build_head(bbox_head)
        
        if rail_head:
            self.rail_head = builder.build_head(rail_head)
        else:
            self.rail_head = None
            
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, points, img_metas=None):
        x = self.backbone(points)
        if self.neck:
            x = self.neck(x)
        return x

    def forward_train(self, points, img_metas, gt_bboxes_3d, gt_labels_3d, gt_poly_3d=None, **kwargs):
        losses = dict()
        
        # 1. 提取特征
        img_feats = self.extract_feat(points, img_metas)
        
        # 2. 检测头前向传播
        outs = self.bbox_head(img_feats)
        
        # [核心修复] 移除 img_metas，只传 3 个参数 (outs, gt_bboxes, gt_labels)
        loss_inputs = (outs, gt_bboxes_3d, gt_labels_3d)
        loss_det = self.bbox_head.loss(*loss_inputs)
        losses.update(loss_det)
        
        # 3. 轨道头前向传播
        if self.rail_head and gt_poly_3d is not None:
            poly_preds = self.rail_head(img_feats)
            loss_poly = self.rail_head.loss(poly_preds, gt_poly_3d)
            losses.update(loss_poly)
            
        return losses

    def simple_test(self, points, img_metas, imgs=None, rescale=False):
        img_feats = self.extract_feat(points, img_metas)
        outs = self.bbox_head(img_feats)
        bbox_list = self.bbox_head.get_bboxes(outs, img_metas, rescale=rescale)
        bbox_results = [
            dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
            for bboxes, scores, labels in bbox_list
        ]
        if self.rail_head:
            poly_preds = self.rail_head(img_feats)
            for i in range(len(bbox_results)):
                bbox_results[i]['rail_preds'] = poly_preds[i].cpu().numpy()
        return bbox_results

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        return self.simple_test(points, img_metas, imgs, rescale)