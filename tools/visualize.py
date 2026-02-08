import argparse
import mmcv
import torch
import numpy as np
import cv2
import os
from mmcv import Config
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset, build_dataloader
from mmcv.runner import load_checkpoint

def project_3d_to_2d(points_3d, lidar2img):
    """
    将3D点投影到图像平面
    points_3d: [N, 3]
    lidar2img: [4, 4]
    """
    # 齐次坐标
    points_h = np.hstack([points_3d, np.ones((len(points_3d), 1))])
    # 投影
    points_2d_h = (lidar2img @ points_h.T).T
    # 归一化 (u/w, v/w)
    points_2d = points_2d_h[:, :2] / points_2d_h[:, 2:3]
    return points_2d

def draw_projected_rail(img, poly_points_3d, lidar2img, color=(0, 255, 0)):
    """绘制投影后的轨道曲线"""
    uv = project_3d_to_2d(poly_points_3d, lidar2img)
    uv = uv.astype(np.int32)
    
    # 过滤图像外的点
    h, w, _ = img.shape
    valid_mask = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    
    # 简单的连线绘制
    for i in range(len(uv) - 1):
        if valid_mask[i] and valid_mask[i+1]:
            cv2.line(img, tuple(uv[i]), tuple(uv[i+1]), color, 2)
    return img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-dir', default='vis_results', help='Output directory')
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.cuda()
    model.eval()

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=1, shuffle=False, dist=False)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    print("Starting Visualization...")
    for i, data in enumerate(data_loader):
        if i >= 10: break # 只看前10张
        
        # 1. 获取图像和标定参数
        # 注意: 这里假设 dataset 返回了 img_metas 包含 lidar2img 矩阵
        img_metas = data['img_metas'][0].data[0][0]
        img_path = img_metas['filename']
        lidar2img = img_metas['lidar2img'] # [4, 4] numpy array
        
        raw_img = cv2.imread(img_path)
        
        # 2. 推理
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)[0]
        
        # 3. 获取预测结果
        # 假设 result 包含 'boxes_3d' 和 'poly_3d' (我们在 RailFusionNet.simple_test 中定义的)
        # 这里需要适配 simple_test 的具体返回格式
        # 示例: pred_polys 是一个 List[numpy array [N, 3]]
        
        # 绘制轨道
        if 'poly_3d' in result:
            pred_polys = result['poly_3d'] # [2, 5, 3]
            for poly in pred_polys:
                # 对控制点进行插值以获得平滑曲线用于可视化
                # ... (此处可增加插值逻辑)
                draw_projected_rail(raw_img, poly, lidar2img, color=(0, 0, 255)) # 红色预测轨道

        # 4. 保存
        filename = os.path.basename(img_path)
        cv2.imwrite(os.path.join(args.out_dir, f'vis_{filename}'), raw_img)
        print(f"Saved {filename}")

if __name__ == '__main__':
    main()