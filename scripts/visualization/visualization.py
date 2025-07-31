"""
一站式GraspNet处理脚本 for ROS2.
功能:
1. 作为ROS2节点启动，连接到实时相机话题。
2. 使用message_filters同步捕获一帧彩色/深度/相机信息。
3. 在内存中直接预处理数据，创建点云。
4. 加载GraspNet模型并执行推理，得到抓取姿态。
5. (可选) 进行碰撞检测。
6. 将最终的场景点云(scene.pcd)和抓取结果(.mat)保存到磁盘。
7. 启动Open3D窗口可视化结果。
8. 完成后自动关闭。
"""
import rclpy
from rclpy.node import Node
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
from graspnetAPI import GraspGroup
import matplotlib.cm as cm
import os
import sys
import numpy as np
import open3d as o3d
import argparse
import scipy.io as scio
from PIL import Image as PilImage # Use a different alias to avoid confusion

import datetime

# 获取当前时间
current_time = datetime.datetime.now()

# 将时间格式化为字符串 (例如: 2023-10-27_15-30-00)
# 这种格式不含特殊字符，适合用于文件名
formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))


import torch
# --- 确保GraspNet相关模块在PYTHONPATH中 ---
# 您可能需要根据您的GraspNet安装路径调整这里
GRASPNet_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # 假设脚本在graspnet-baseline/的子目录
sys.path.append(GRASPNet_ROOT)
sys.path.append(os.path.join(GRASPNet_ROOT, 'models'))
sys.path.append(os.path.join(GRASPNet_ROOT, 'utils'))

from graspnetAPI import GraspGroup
from graspnet import GraspNet, pred_decode
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo as GraspNetCameraInfo, create_point_cloud_from_depth_image

class GraspNetProcessorNode(Node):
    def __init__(self, cfgs):
        super().__init__('graspnet_processor_node')
        self.cfgs = cfgs
        self.bridge = CvBridge()
        self.processed = False

        # 1. 初始化GraspNet模型 (只加载一次)
        self.get_logger().info("正在加载GraspNet模型...")
        self.net = self._get_net()
        self.get_logger().info("模型加载成功！")

        # 2. 设置ROS订阅者
        color_topic = "/camera/color/image_raw"
        depth_topic = "/camera/depth_registered/image_rect"
        info_topic = "/camera/color/camera_info"
        
        self.color_sub = message_filters.Subscriber(self, Image, color_topic)
        self.depth_sub = message_filters.Subscriber(self, Image, depth_topic)
        self.info_sub = message_filters.Subscriber(self, CameraInfo, info_topic)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub, self.info_sub], queue_size=5, slop=0.2
        )
        self.ts.registerCallback(self.process_callback)
        self.get_logger().info("节点已启动，正在等待同步的相机数据...")

    def _get_net(self):
        net = GraspNet(input_feature_dim=0, num_view=self.cfgs.num_view, num_angle=12, num_depth=4,
                cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)

        script_path = os.path.realpath(__file__)
        folder_path = os.path.dirname(script_path)
        path = os.path.join(folder_path, "checkpoint-rs.tar")
        checkpoint = torch.load(path)
        net.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        self.get_logger().info(f"-> loaded checkpoint '{path}' (epoch: {start_epoch})")
        net.eval()
        return net

    def process_callback(self, color_msg, depth_msg, info_msg):
        if self.processed:
            return

        self.get_logger().info("接收到同步帧，开始处理...")
        self.processed = True

        # --- 第1步: 将ROS消息转换为Numpy数据 ---
        color_image_bgr = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
        color_image_rgb = color_image_bgr[:, :, ::-1].copy() # BGR to RGB
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
        
        intrinsic_matrix = np.array(info_msg.k).reshape(3, 3)
        factor_depth = 1000.0 

        # --- 第2步: 数据预处理 (与demo.py逻辑类似，但在内存中完成) ---
        # 创建工作区掩码 (可根据需要调整)
        height, width = depth_image.shape
        
        # <--- 新增的修复代码 START --->
        # 在这里，我们将彩色图像的尺寸调整为与深度图像完全一致
        color_image_rgb = cv2.resize(color_image_rgb, (width, height))
        self.get_logger().info(f"彩色图像已从 {color_msg.height}x{color_msg.width} 缩放至 {height}x{width} 以匹配深度图。")
        # <--- 新增的修复代码 END --->

        workspace_mask = np.zeros((height, width), dtype=np.uint8)
        workspace_mask[int(height*0.15):int(height*0.85), int(width*0.15):int(width*0.85)] = 1
        
        # 创建点云
        camera = GraspNetCameraInfo(width, height, intrinsic_matrix[0,0], intrinsic_matrix[1,1], intrinsic_matrix[0,2], intrinsic_matrix[1,2], factor_depth)
        cloud = create_point_cloud_from_depth_image(depth_image, camera, organized=True)

        # 应用掩码
        # 现在 color_image_rgb 和 mask 的尺寸完全相同，不会再报错
        mask = (workspace_mask > 0) & (depth_image > 0)
        cloud_masked = cloud[mask]
        color_masked = color_image_rgb[mask] / 255.0

        # 采样点
        if len(cloud_masked) == 0:
            self.get_logger().error("工作区内没有有效的点云数据！请检查相机视野或掩码设置。")
            self.shutdown()
            return

        if len(cloud_masked) >= self.cfgs.num_point:
            idxs = np.random.choice(len(cloud_masked), self.cfgs.num_point, replace=False)
        else:
            idxs = np.concatenate([np.arange(len(cloud_masked)), np.random.choice(len(cloud_masked), self.cfgs.num_point - len(cloud_masked), replace=True)])
        cloud_sampled = cloud_masked[idxs]
        
        # 准备输入给网络的end_points字典
        end_points = dict()
        cloud_sampled_torch = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        end_points['point_clouds'] = cloud_sampled_torch.to(device)
        
        # 创建用于保存和可视化的完整点云对象
        scene_cloud = o3d.geometry.PointCloud()
        scene_cloud.points = o3d.utility.Vector3dVector(cloud_masked)
        scene_cloud.colors = o3d.utility.Vector3dVector(color_masked)

        # ... 后面的代码保持不变 ...
        # --- 第3步: 运行GraspNet推理 ---
        self.get_logger().info("正在执行抓取姿态推理...")
        with torch.no_grad():
            end_points_res = self.net(end_points)
            grasp_preds = pred_decode(end_points_res)
        gg_array = grasp_preds[0].detach().cpu().numpy()
        gg = GraspGroup(gg_array)
        self.get_logger().info(f"推理完成，初步检测到 {len(gg)} 个抓取。")
        
        # --- 第4步: 碰撞检测 ---
        if self.cfgs.collision_thresh > 0:
            self.get_logger().info("正在进行碰撞检测...")
            mfcdetector = ModelFreeCollisionDetector(cloud_masked, voxel_size=self.cfgs.voxel_size)
            collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=self.cfgs.collision_thresh)
            gg = gg[~collision_mask]
            self.get_logger().info(f"碰撞检测后剩余 {len(gg)} 个抓取。")

        # --- 第5步: 保存结果 ---
        script_path = os.path.realpath(__file__)
        folder_path = os.path.dirname(script_path)
        self.output_dir = os.path.join(folder_path, f"visualization_output/graspnet_ros_output_{formatted_time}")
        # self.output_dir = f"visualization_output/graspnet_ros_output_{formatted_time}"
        self.get_logger().info(f"正在将结果保存到 '{self.output_dir}' 目录...")
        self._save_results(gg, scene_cloud, intrinsic_matrix, factor_depth, self.output_dir)

        # --- 第6步: 可视化结果 ---
        self.get_logger().info("正在启动可视化窗口... (关闭窗口后程序将退出)")
        self._vis_grasps(gg, scene_cloud)

        # --- 第7步: 完成并关闭 ---
        self.get_logger().info("所有流程执行完毕！")
        self.shutdown()
    def _save_results(self, gg, cloud, intrinsic, factor_depth, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        pcd_path = os.path.join(output_dir, 'scene.pcd')
        o3d.io.write_point_cloud(pcd_path, cloud)
        self.get_logger().info(f"-> 点云已保存至: {pcd_path}")

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
        self.get_logger().info(f"-> .mat 结果文件已保存至: {mat_path}")

    def _vis_grasps(self, gg, cloud):
        """
        根据抓取分数使用颜色映射来可视化抓取。
        """
        gg.nms()
        gg.sort_by_score()
        gg_top_k = gg[:50] # 最多显示50个

        # 如果没有抓取，则直接显示点云
        if len(gg_top_k) == 0:
            o3d.visualization.draw_geometries([cloud])
            return

        # 准备颜色映射
        scores = [g.score for g in gg_top_k]
        # 从self.cfgs中读取颜色映射表的名称
        cmap = cm.get_cmap(self.cfgs.cmap) 

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

        o3d.visualization.draw_geometries([cloud, *grippers])

    def shutdown(self):
        self.get_logger().info('正在关闭节点...')
        self.destroy_node()
        # rclpy.shutdown() # 让主函数来关闭

def main():
    # --- 使用argparse解析GraspNet的参数 ---
    parser = argparse.ArgumentParser(description="一站式GraspNet处理脚本 for ROS2")
    parser.add_argument('--checkpoint_path', help='GraspNet模型检查点路径')
    parser.add_argument('--num_point', type=int, default=20000, help='采样点数量')
    parser.add_argument('--num_view', type=int, default=300, help='视角数量')
    parser.add_argument('--collision_thresh', type=float, default=0.01, help='碰撞检测阈值')
    parser.add_argument('--voxel_size', type=float, default=0.01, help='体素大小')
    parser.add_argument('--cmap', type=str, default='viridis',
                        help="用于可视化分数的颜色映射表 (例如: viridis, plasma, jet)。")
    # parser.add_argument('--output_dir', type=str, default='graspnet_ros_output', help='保存结果的目录')
    
    # 解析参数
    cfgs, ros_args = parser.parse_known_args()

    # --- 初始化并运行ROS节点 ---
    rclpy.init(args=ros_args)
    processor_node = GraspNetProcessorNode(cfgs)
    try:
        rclpy.spin(processor_node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()