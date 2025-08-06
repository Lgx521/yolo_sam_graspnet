#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np
import open3d as o3d
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point
import cv2
import sys
from typing import Optional

# 假设您的 cv_segmentation.py 在这个路径下
sys.path.append('/home/roar/graspnet/graspnet-baseline/kinova_graspnet_ros2/utils')
from cv_segmentation import segment_objects

# 导入我们创建的服务
from kinova_graspnet_ros2.srv import GenerateObstacles

# 导入与 grasp_detection_service.py 相同的点云创建工具
sys.path.append('/home/roar/graspnet/graspnet-baseline/utils')
from data_utils import CameraInfo as GraspNetCameraInfo, create_point_cloud_from_depth_image


class ObstacleGeometryNode(Node):
    """
    一个ROS2节点，用于识别除目标物体外的障碍物，
    并将其几何形状建模为凸包进行可视化。
    采用鲁棒的点云减法策略，并集成了性能和精度优化。
    """
    def __init__(self):
        super().__init__('obstacle_geometry_node')

        # 声明与抓取检测服务一致的参数
        self.declare_parameter('color_image_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_image_topic', '/camera/depth_registered/image_rect')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('depth_camera_frame', 'camera_depth_frame')

        # 获取参数
        color_topic = self.get_parameter('color_image_topic').value
        depth_topic = self.get_parameter('depth_image_topic').value
        info_topic = self.get_parameter('camera_info_topic').value
        self.camera_frame = self.get_parameter('depth_camera_frame').value

        # 初始化工具
        self.bridge = CvBridge()

        # 数据存储
        self.latest_color_image: Optional[Image] = None
        self.latest_depth_image: Optional[Image] = None
        self.latest_camera_info: Optional[CameraInfo] = None
        self.camera_data_ready = False

        # 订阅相机话题
        self.color_sub = self.create_subscription(Image, color_topic, self.color_callback, 10)
        self.depth_sub = self.create_subscription(Image, depth_topic, self.depth_callback, 10)
        self.info_sub = self.create_subscription(CameraInfo, info_topic, self.info_callback, 10)

        # 发布障碍物可视化Marker
        self.marker_pub = self.create_publisher(MarkerArray, 'obstacle_markers', 10)
        
        # 创建服务，当被调用时执行障碍物检测
        self.obstacle_service = self.create_service(
            GenerateObstacles,
            'generate_obstacles',
            self.generate_obstacles_callback
        )

        self.get_logger().info('Obstacle Geometry Node 已初始化，等待相机数据和服务调用...')

    def color_callback(self, msg: Image):
        self.latest_color_image = msg
        self.check_data_ready()

    def depth_callback(self, msg: Image):
        self.latest_depth_image = msg
        self.check_data_ready()

    def info_callback(self, msg: CameraInfo):
        self.latest_camera_info = msg
        self.check_data_ready()

    def check_data_ready(self):
        if not self.camera_data_ready:
            if self.latest_color_image and self.latest_depth_image and self.latest_camera_info:
                self.camera_data_ready = True
                self.get_logger().info('相机数据已准备就绪。')

    def generate_obstacles_callback(self, request: GenerateObstacles.Request, response: GenerateObstacles.Response):
        """服务回调函数，执行障碍物检测和凸包生成"""
        if not self.camera_data_ready:
            response.success = False
            response.message = "相机数据尚未准备好。"
            self.get_logger().warn(response.message)
            return response

        self.get_logger().info(f"收到障碍物生成请求，目标物体为: '{request.target_object_class}'")

        try:
            color_cv = self.bridge.imgmsg_to_cv2(self.latest_color_image, "rgb8")
            depth_cv = self.bridge.imgmsg_to_cv2(self.latest_depth_image, "passthrough")
            cam_info = self.latest_camera_info
            fx, fy, cx, cy = cam_info.k[0], cam_info.k[4], cam_info.k[2], cam_info.k[5]
            graspnet_cam = GraspNetCameraInfo(depth_cv.shape[1], depth_cv.shape[0], fx, fy, cx, cy, 1000.0)

            # --- 核心逻辑：点云减法 ---

            # 步骤 1: 生成整个场景在工作空间内的点云
            self.get_logger().info("步骤 1/5: 生成场景完整点云并过滤...")
            
            full_cloud_organized = create_point_cloud_from_depth_image(depth_cv, graspnet_cam, organized=True)
            valid_depth_mask = (depth_cv > 0)
            
            scene_pcd = o3d.geometry.PointCloud()
            scene_pcd.points = o3d.utility.Vector3dVector(full_cloud_organized[valid_depth_mask])

            min_bound = np.array([-0.5, -0.4, 0.1])
            max_bound = np.array([0.5, 0.4, 1.0])
            bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
            workspace_pcd = scene_pcd.crop(bbox)
            self.get_logger().info(f"  - 工作空间内总点数: {len(workspace_pcd.points)}")

            # 步骤 2: 分割目标物体，并对掩码进行膨胀处理
            self.get_logger().info(f"步骤 2/5: 分割目标物体 '{request.target_object_class}'...")
            target_mask, _ = segment_objects(color_cv, target_class=request.target_object_class, return_vis=True)
            
            if target_mask is None:
                self.get_logger().warn(f"无法分割目标物体 '{request.target_object_class}'。")
                target_mask = np.zeros(color_cv.shape[:2], dtype=np.uint8)

            # --- 新增：对目标掩码进行膨胀处理，以创建安全边界 ---
            kernel_size = 7 # 可以调整内核大小, 5或7通常不错
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            dilated_target_mask = cv2.dilate(target_mask, kernel, iterations=1)
            # --- 膨胀处理结束 ---

            if dilated_target_mask.shape != depth_cv.shape:
                target_mask_resized = cv2.resize(dilated_target_mask, (depth_cv.shape[1], depth_cv.shape[0]), interpolation=cv2.INTER_NEAREST)
            else:
                target_mask_resized = dilated_target_mask
            
            final_target_mask = (target_mask_resized > 0) & valid_depth_mask
            target_pcd = o3d.geometry.PointCloud()
            target_pcd.points = o3d.utility.Vector3dVector(full_cloud_organized[final_target_mask])
            self.get_logger().info(f"  - (膨胀后)目标物体点数: {len(target_pcd.points)}")

            # 步骤 3: 计算障碍物点云
            self.get_logger().info("步骤 3/5: 计算障碍物点云...")
            
            if len(target_pcd.points) > 0 and len(workspace_pcd.points) > 0:
                # --- 性能优化：使用Open3D内置函数代替Python循环 ---
                distances = workspace_pcd.compute_point_cloud_distance(target_pcd)
                obstacle_indices = np.where(np.asarray(distances) > 0.005)[0] # 阈值提高到5mm
                obstacle_pcd = workspace_pcd.select_by_index(obstacle_indices)
                # --- 优化结束 ---
            else:
                obstacle_pcd = workspace_pcd
            
            self.get_logger().info(f"  - 计算后的障碍物点数: {len(obstacle_pcd.points)}")
            
            pcd = obstacle_pcd 
            
            if len(pcd.points) < 20: 
                 self.get_logger().info("过滤后点数过少，无障碍物需要处理。")
                 self.clear_markers()
                 response.success = True
                 response.message = "过滤后无障碍物。"
                 return response
            
            self.get_logger().info("  - 进行体素下采样...")
            voxel_size = 0.01 
            pcd = pcd.voxel_down_sample(voxel_size)
            self.get_logger().info(f"  - 体素下采样后点数: {len(pcd.points)}")
                 
            self.get_logger().info(f"步骤 4/5: 对 {len(pcd.points)} 个障碍物点进行聚类...")
            labels = np.array(pcd.cluster_dbscan(eps=0.04, min_points=20, print_progress=True))
            
            max_label = labels.max()
            noise_points = np.sum(np.array(labels) == -1)
            self.get_logger().info(f"聚类完成，发现 {max_label + 1} 个障碍物。另外有 {noise_points} 个点被认为是噪声。")
            
            self.get_logger().info("步骤 5/5: 生成凸包并发布可视化标记...")
            marker_array = MarkerArray()
            
            clear_marker = Marker()
            clear_marker.header.frame_id = self.camera_frame
            clear_marker.action = Marker.DELETEALL
            marker_array.markers.append(clear_marker)

            for i in range(max_label + 1):
                cluster_indices = np.where(labels == i)[0]
                if len(cluster_indices) == 0: continue
                
                cluster_pcd = pcd.select_by_index(cluster_indices)
                
                try:
                    hull, _ = cluster_pcd.compute_convex_hull()
                    if len(np.asarray(hull.vertices)) > 0:
                        hull.orient_triangles()
                        hull_marker = self.create_hull_marker(hull, i)
                        marker_array.markers.append(hull_marker)
                except Exception as e:
                    self.get_logger().warn(f"为聚类 {i} 创建凸包失败: {e}")

            self.marker_pub.publish(marker_array)
            response.success = True
            response.message = f"成功生成并发布了 {max_label + 1} 个障碍物的凸包。"
            self.get_logger().info(response.message)

        except Exception as e:
            self.get_logger().error(f"处理过程中发生错误: {e}")
            import traceback
            self.get_logger().error(f"错误追溯: {traceback.format_exc()}")
            response.success = False
            response.message = str(e)
        return response
    
    def create_hull_marker(self, hull: o3d.geometry.TriangleMesh, marker_id: int) -> Marker:
        # ... (此函数与上一版相同，无需修改)
        marker = Marker()
        marker.header.frame_id = self.camera_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "obstacle_hulls"
        marker.id = marker_id
        marker.type = Marker.TRIANGLE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        marker.color = ColorRGBA(r=0.5, g=0.0, b=0.8, a=0.4)
        vertices = np.asarray(hull.vertices)
        triangles = np.asarray(hull.triangles)
        for triangle_indices in triangles:
            p1 = Point(); p1.x, p1.y, p1.z = vertices[triangle_indices[0]]; marker.points.append(p1)
            p2 = Point(); p2.x, p2.y, p2.z = vertices[triangle_indices[1]]; marker.points.append(p2)
            p3 = Point(); p3.x, p3.y, p3.z = vertices[triangle_indices[2]]; marker.points.append(p3)
        return marker

    def clear_markers(self):
        # ... (此函数与上一版相同，无需修改)
        marker_array = MarkerArray()
        clear_marker = Marker()
        clear_marker.header.frame_id = self.camera_frame
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)
        self.marker_pub.publish(marker_array)
        self.get_logger().info("已清除所有障碍物标记。")

def main(args=None):
    # ... (此函数与上一版相同，无需修改)
    rclpy.init(args=args)
    try:
        node = ObstacleGeometryNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if 'node' in locals() and rclpy.ok():
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()