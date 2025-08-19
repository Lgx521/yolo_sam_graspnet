"""
一站式GraspNet处理脚本 (文件输入版 - 参数内嵌).
功能:
1. 在代码内部直接指定彩色/深度图像路径和相机内参。
2. 在内存中直接预处理数据，创建点云。
3. 加载GraspNet模型并执行推理，得到抓取姿态。
4. (可选) 进行碰撞检测。
5. 将最终的场景点云(scene.pcd)和抓取结果(.mat)保存到磁盘。
6. 启动Open3D窗口可视化结果。
7. 完成后自动关闭。
"""
import os
import sys
import numpy as np
import open3d as o3d
import scipy.io as scio
import torch
import cv2
import datetime
import matplotlib.cm as cm
from types import SimpleNamespace # 用于创建一个简单的对象来存储配置

# --- 确保GraspNet相关模块在PYTHONPATH中 ---
# 您可能需要根据您的GraspNet安装路径调整这里
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnetAPI import GraspGroup
from graspnet import GraspNet, pred_decode
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo as GraspNetCameraInfo, create_point_cloud_from_depth_image

def get_net(cfgs):
    """加载GraspNet模型"""
    net = GraspNet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
            cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    # 优先使用配置中指定的路径，否则使用默认路径
    checkpoint_path = cfgs.checkpoint_path
    if not checkpoint_path:
        script_path = os.path.realpath(__file__)
        folder_path = os.path.dirname(script_path)
        checkpoint_path = os.path.join(folder_path, "checkpoint-rs.tar")

    if not os.path.exists(checkpoint_path):
        print(f"错误: 找不到模型权重文件 '{checkpoint_path}'")
        sys.exit(1)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    print(f"-> 已加载模型 '{checkpoint_path}' (epoch: {start_epoch})")
    net.eval()
    return net, device

def save_results(gg, cloud, intrinsic, factor_depth, output_dir):
    """保存点云和抓取结果"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    pcd_path = os.path.join(output_dir, 'scene.pcd')
    o3d.io.write_point_cloud(pcd_path, cloud)
    print(f"-> 点云已保存至: {pcd_path}")

    # NMS和排序
    gg.nms()
    gg.sort_by_score()
    
    poses_list, scores_list, widths_list = [], [], []
    for grasp in gg:
        pose_matrix = np.hstack([grasp.rotation_matrix, grasp.translation.reshape(3, 1)])
        poses_list.append(pose_matrix)
        scores_list.append(grasp.score)
        widths_list.append(grasp.width)

    if len(poses_list) > 0:
        poses_array = np.stack(poses_list, axis=-1)
    else:
        poses_array = np.zeros((3, 4, 0))

    mat_dict = {
        'poses': poses_array.astype(np.float32),
        'scores': np.array(scores_list, dtype=np.float32),
        'widths': np.array(widths_list, dtype=np.float32),
        'intrinsic_matrix': intrinsic,
        'factor_depth': np.array([[factor_depth]])
    }
    mat_path = os.path.join(output_dir, 'grasp_results.mat')
    scio.savemat(mat_path, mat_dict)
    print(f"-> .mat 结果文件已保存至: {mat_path}")

def vis_grasps(gg, cloud, cfgs):
    """根据抓取分数使用颜色映射来可视化抓取"""
    gg.nms()
    gg.sort_by_score()
    gg_top_k = gg[:50] # 最多显示50个

    # 如果没有抓取，则直接显示点云
    if len(gg_top_k) == 0:
        print("未检测到有效抓取，仅显示场景点云。")
        o3d.visualization.draw_geometries([cloud])
        return

    # 准备颜色映射
    scores = [g.score for g in gg_top_k]
    cmap = cm.get_cmap(cfgs.cmap) 

    # 归一化分数
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        norm_scores = [1.0] * len(scores)
    else:
        norm_scores = [(s - min_score) / (max_score - min_score) for s in scores]

    # 为每个抓取创建带颜色的几何模型
    grippers = []
    for i, grasp in enumerate(gg_top_k):
        # 将归一化的分数映射到颜色
        color = cmap(norm_scores[i])[:3] # 取RGB部分，忽略Alpha
        grippers.append(grasp.to_open3d_geometry(color=color))

    print(f"正在可视化场景点云和 {len(grippers)} 个最佳抓取姿态...")
    o3d.visualization.draw_geometries([cloud, *grippers])

def run_inference(cfgs):
    """主推理流程"""
    # 1. 加载模型
    print("正在加载GraspNet模型...")
    net, device = get_net(cfgs)
    print("模型加载成功！")

    # 2. 加载图像
    print(f"正在从文件加载图像: \n  - 彩色: {cfgs.color_path}\n  - 深度: {cfgs.depth_path}")
    color_image_bgr = cv2.imread(cfgs.color_path, cv2.IMREAD_COLOR)
    # 假设深度图像是16位单通道图像 (e.g., in mm)
    depth_image = cv2.imread(cfgs.depth_path, cv2.IMREAD_UNCHANGED)

    if color_image_bgr is None:
        print(f"错误: 无法加载彩色图像 '{cfgs.color_path}'")
        return
    if depth_image is None:
        print(f"错误: 无法加载深度图像 '{cfgs.depth_path}'")
        return
        
    color_image_rgb = cv2.cvtColor(color_image_bgr, cv2.COLOR_BGR2RGB)

    # 3. 构建相机参数
    height, width, _ = color_image_rgb.shape
    depth_height, depth_width = depth_image.shape
    
    # 确保彩色图和深度图尺寸一致
    if (height, width) != (depth_height, depth_width):
        print(f"警告: 彩色图像 ({height}x{width}) 和深度图像 ({depth_height}x{depth_width}) 尺寸不匹配。")
        print(f"将彩色图像缩放至 {depth_height}x{depth_width} 以匹配深度图。")
        color_image_rgb = cv2.resize(color_image_rgb, (depth_width, depth_height))
        height, width = depth_height, depth_width
    
    intrinsic_matrix = np.array([[cfgs.fx, 0, cfgs.cx], [0, cfgs.fy, cfgs.cy], [0, 0, 1]])
    camera = GraspNetCameraInfo(width, height, cfgs.fx, cfgs.fy, cfgs.cx, cfgs.cy, cfgs.factor_depth)

    # 4. 数据预处理和点云创建
    print("正在创建点云...")
    # 创建工作区掩码 (可根据需要调整)
    workspace_mask = np.zeros((height, width), dtype=np.uint8)
    workspace_mask[int(height*0.15):int(height*0.85), int(width*0.15):int(width*0.85)] = 1
    
    # 创建点云
    cloud = create_point_cloud_from_depth_image(depth_image, camera, organized=True)

    # 应用掩码
    mask = (workspace_mask > 0) & (depth_image > 0)
    cloud_masked = cloud[mask]
    color_masked = color_image_rgb[mask] / 255.0

    # 采样点
    if len(cloud_masked) == 0:
        print("错误: 工作区内没有有效的点云数据！请检查相机视野或掩码设置。")
        return

    if len(cloud_masked) >= cfgs.num_point:
        idxs = np.random.choice(len(cloud_masked), cfgs.num_point, replace=False)
    else:
        idxs = np.concatenate([np.arange(len(cloud_masked)), np.random.choice(len(cloud_masked), cfgs.num_point - len(cloud_masked), replace=True)])
    cloud_sampled = cloud_masked[idxs]
    
    # 准备输入给网络的end_points字典
    end_points = dict()
    cloud_sampled_torch = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
    end_points['point_clouds'] = cloud_sampled_torch.to(device)
    
    # 创建用于保存和可视化的完整点云对象
    scene_cloud = o3d.geometry.PointCloud()
    scene_cloud.points = o3d.utility.Vector3dVector(cloud_masked)
    scene_cloud.colors = o3d.utility.Vector3dVector(color_masked)
    print(f"点云创建完成，共采样 {len(cloud_sampled)} 个点。")

    # 5. 运行GraspNet推理
    print("正在执行抓取姿态推理...")
    with torch.no_grad():
        end_points_res = net(end_points)
        grasp_preds = pred_decode(end_points_res)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(gg_array)
    print(f"推理完成，初步检测到 {len(gg)} 个抓取。")
    
    # 6. 碰撞检测
    if cfgs.collision_thresh > 0:
        print("正在进行碰撞检测...")
        mfcdetector = ModelFreeCollisionDetector(cloud_masked, voxel_size=cfgs.voxel_size)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
        gg = gg[~collision_mask]
        print(f"碰撞检测后剩余 {len(gg)} 个抓取。")

    # 7. 保存结果
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(os.path.dirname(__file__), f"visualization_output/graspnet_output_{current_time}")
    print(f"正在将结果保存到 '{output_dir}' 目录...")
    save_results(gg, scene_cloud, intrinsic_matrix, cfgs.factor_depth, output_dir)

    # 8. 可视化结果
    print("正在启动可视化窗口... (关闭窗口后程序将退出)")
    vis_grasps(gg, scene_cloud, cfgs)

    print("所有流程执行完毕！")

def main():
    # ==========================================================================
    # --- 用户配置区 ---
    # 请在这里修改您的文件路径和相机参数
    # ==========================================================================
    cfgs = SimpleNamespace(
        # 1. 输入数据路径 (请使用绝对路径或相对于此脚本的路径)
        color_path = "path/to/your/color_image.png", # 例如: "/home/user/data/scene1/color.png"
        depth_path = "path/to/your/depth_image.png", # 例如: "/home/user/data/scene1/depth.png"

        # 2. 相机内参 (请替换为您相机的值)
        fx = 615.8,  # X轴焦距
        fy = 615.8,  # Y轴焦距
        cx = 320.0,  # X轴主点
        cy = 240.0,  # Y轴主点
        factor_depth = 1000.0, # 深度图像的缩放因子。如果深度图中1000代表1米，则此值为1000.0

        # 3. GraspNet模型和推理参数 (通常无需修改)
        # 模型权重路径。设置为None将自动加载与此脚本同目录下的 "checkpoint-rs.tar" 文件。
        # 如有需要，可指定绝对路径: "/path/to/your/checkpoint-rs.tar"
        checkpoint_path = None,
        
        num_point = 20000,          # 采样点数量
        num_view = 300,             # 视角数量
        collision_thresh = 0.01,    # 碰撞检测阈值 (设置为0则禁用)
        voxel_size = 0.01,          # 体素大小 (用于碰撞检测)
        cmap = 'viridis'            # 可视化抓取分数的颜色映射表
    )
    # ==========================================================================
    # --- 配置区结束 ---
    # ==========================================================================

    # 检查路径是否有效
    if not os.path.exists(cfgs.color_path) or not os.path.exists(cfgs.depth_path):
        print("="*50)
        print(f"错误: 找不到图像文件！")
        print(f"  - 检查彩色图像路径: {os.path.abspath(cfgs.color_path)}")
        print(f"  - 检查深度图像路径: {os.path.abspath(cfgs.depth_path)}")
        print("请确保在代码的 '用户配置区' 中设置了正确的文件路径。")
        print("="*50)
        return

    run_inference(cfgs)

if __name__ == '__main__':
    main()