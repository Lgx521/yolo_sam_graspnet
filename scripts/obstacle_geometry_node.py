#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np
import open3d as o3d
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point, Pose
import cv2
import sys
from typing import Optional
import time

# --- 新增: 导入MoveIt相关的库 ---
import moveit_commander
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import Mesh, MeshTriangle

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
    并将其几何形状建模为Alpha Shape，然后添加到MoveIt规划场景中。
    """
    def __init__(self):
        super().__init__('obstacle_geometry_node')

        # 声明与抓取检测服务一致的参数 (遵循您的版本)
        self.declare_parameter('color_image_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_image_topic', '/camera/depth_registered/image_rect')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('depth_camera_frame', 'camera_depth_frame')
        # --- 新增: 声明MoveIt所需的base_frame ---
        self.declare_parameter('base_frame', 'base_link')

        # 获取参数
        color_topic = self.get_parameter('color_image_topic').value
        depth_topic = self.get_parameter('depth_image_topic').value
        info_topic = self.get_parameter('camera_info_topic').value
        self.camera_frame = self.get_parameter('depth_camera_frame').value
        self.base_frame = self.get_parameter('base_frame').value

        # --- 新增: 初始化MoveIt Commander和Planning Scene Interface ---
        try:
            self.scene = moveit_commander.PlanningSceneInterface()
            self.get_logger().info("MoveIt Planning Scene Interface 初始化成功。")
        except Exception as e:
            self.get_logger().error(f"MoveIt Planning Scene Interface 初始化失败: {e}")
            self.scene = None
        # --- 初始化结束 ---

        # 初始化工具 (遵循您的版本)
        self.bridge = CvBridge()
        self.latest_color_image: Optional[Image] = None
        self.latest_depth_image: Optional[Image] = None
        self.latest_camera_info: Optional[CameraInfo] = None
        self.camera_data_ready = False

        # 订阅相机话题 (遵循您的版本)
        self.color_sub = self.create_subscription(Image, color_topic, self.color_callback, 10)
        self.depth_sub = self.create_subscription(Image, depth_topic, self.depth_callback, 10)
        self.info_sub = self.create_subscription(Image, info_topic, self.info_callback, 10)

        # 发布障碍物可视化Marker
        self.marker_pub = self.create_publisher(MarkerArray, 'obstacle_markers', 10)
        
        # 创建服务，当被调用时执行障碍物检测
        self.obstacle_service = self.create_service(
            GenerateObstacles,
            'generate_obstacles',
            self.generate_obstacles_callback
        )

        self.get_logger().info('Obstacle Geometry Node 已初始化，等待相机数据和服务调用...')

    # 相机回调函数 (与您的版本相同)
    def color_callback(self, msg: Image): self.latest_color_image = msg; self.check_data_ready()
    def depth_callback(self, msg: Image): self.latest_depth_image = msg; self.check_data_ready()
    def info_callback(self, msg: CameraInfo): self.latest_camera_info = msg; self.check_data_ready()
    def check_data_ready(self):
        if not self.camera_data_ready:
            if self.latest_color_image and self.latest_depth_image and self.latest_camera_info:
                self.camera_data_ready = True
                self.get_logger().info('相机数据已准备就绪。')

    def generate_obstacles_callback(self, request: GenerateObstacles.Request, response: GenerateObstacles.Response):
        """服务回调函数，执行障碍物检测和凸包生成"""
        # --- 修改: 检查self.scene是否已成功初始化 ---
        if not self.camera_data_ready or self.scene is None:
            response.success = False
            response.message = "相机或MoveIt场景尚未准备好。"
            self.get_logger().warn(response.message)
            return response
            
        self.get_logger().info(f"收到障碍物生成请求，目标物体为: '{request.target_object_class}'")

        # --- 新增: 在开始时清除旧的障碍物 ---
        self.clear_all_obstacles()
        # --- 清除结束 ---

        try:
            # 核心算法逻辑 (与您的版本相同)
            color_cv = self.bridge.imgmsg_to_cv2(self.latest_color_image, "rgb8")
            depth_cv = self.bridge.imgmsg_to_cv2(self.latest_depth_image, "passthrough")
            cam_info = self.latest_camera_info
            fx, fy, cx, cy = cam_info.k[0], cam_info.k[4], cam_info.k[2], cam_info.k[5]
            graspnet_cam = GraspNetCameraInfo(depth_cv.shape[1], depth_cv.shape[0], fx, fy, cx, cy, 1000.0)
            
            self.get_logger().info("步骤 1/5: 生成场景完整点云并过滤...")
            full_cloud_organized = create_point_cloud_from_depth_image(depth_cv, graspnet_cam, organized=True)
            valid_depth_mask = (depth_cv > 0)
            scene_pcd = o3d.geometry.PointCloud()
            scene_pcd.points = o3d.utility.Vector3dVector(full_cloud_organized[valid_depth_mask])
            min_bound = np.array([-0.3, -0.3, 0.05])
            max_bound = np.array([0.3, 0.3, 0.8])
            workspace_pcd = scene_pcd.crop(o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound))
            self.get_logger().info(f"  - 工作空间内总点数: {len(workspace_pcd.points)}")

            self.get_logger().info(f"步骤 2/5: 分割目标物体 '{request.target_object_class}'...")
            target_mask, _ = segment_objects(color_cv, target_class=request.target_object_class, return_vis=True)
            if target_mask is None:
                target_mask = np.zeros(color_cv.shape[:2], dtype=np.uint8)
            kernel_size = 13
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            dilated_target_mask = cv2.dilate(target_mask, kernel, iterations=1)
            if dilated_target_mask.shape != depth_cv.shape:
                target_mask_resized = cv2.resize(dilated_target_mask, (depth_cv.shape[1], depth_cv.shape[0]), interpolation=cv2.INTER_NEAREST)
            else:
                target_mask_resized = dilated_target_mask
            final_target_mask = (target_mask_resized > 0) & valid_depth_mask
            target_pcd = o3d.geometry.PointCloud()
            target_pcd.points = o3d.utility.Vector3dVector(full_cloud_organized[final_target_mask])
            self.get_logger().info(f"  - (膨胀后)目标物体点数: {len(target_pcd.points)}")

            self.get_logger().info("步骤 3/5: 计算障碍物点云...")
            if len(target_pcd.points) > 0 and len(workspace_pcd.points) > 0:
                distances = workspace_pcd.compute_point_cloud_distance(target_pcd)
                obstacle_indices = np.where(np.asarray(distances) > 0.005)[0]
                obstacle_pcd = workspace_pcd.select_by_index(obstacle_indices)
            else:
                obstacle_pcd = workspace_pcd
            self.get_logger().info(f"  - 计算后的障碍物点数: {len(obstacle_pcd.points)}")
            
            pcd = obstacle_pcd 
            if len(pcd.points) < 20: 
                 self.get_logger().info("过滤后点数过少，无障碍物需要处理。"); self.clear_all_obstacles(); response.success = True; response.message = "过滤后无障碍物。"; return response
            
            self.get_logger().info("  - 进行体素下采样...")
            voxel_size = 0.01 
            pcd = pcd.voxel_down_sample(voxel_size)
            self.get_logger().info(f"  - 体素下采样后点数: {len(pcd.points)}")
                 
            self.get_logger().info(f"步骤 4/5: 对 {len(pcd.points)} 个障碍物点进行聚类...")
            labels = np.array(pcd.cluster_dbscan(eps=0.04, min_points=30, print_progress=True))
            max_label = labels.max()
            self.get_logger().info(f"聚类完成，发现 {max_label + 1} 个障碍物。")

            # --- 核心修改: 不仅发布Marker，还要添加到Planning Scene ---
            self.get_logger().info("步骤 5/5: 生成网格，发布可视化标记并添加到规划场景...")
            marker_array = MarkerArray()
            alpha_value = 0.05
            obstacle_count = 0
            for i in range(max_label + 1):
                cluster_indices = np.where(labels == i)[0]
                if len(cluster_indices) < 4: continue
                
                cluster_pcd = pcd.select_by_index(cluster_indices)
                try:
                    alpha_shape_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(cluster_pcd, alpha_value)
                    alpha_shape_mesh.compute_vertex_normals()
                    alpha_shape_mesh.orient_triangles()

                    # 添加到规划场景
                    obstacle_name = f"obstacle_{i}"
                    self.add_mesh_to_planning_scene(obstacle_name, alpha_shape_mesh)
                    
                    # 发布可视化Marker
                    mesh_marker = self.create_mesh_marker(alpha_shape_mesh, i)
                    marker_array.markers.append(mesh_marker)
                    obstacle_count += 1
                except Exception as e:
                    self.get_logger().warn(f"为聚类 {i} 创建或添加网格失败: {e}")

            self.marker_pub.publish(marker_array)
            msg = f"成功将 {obstacle_count} 个障碍物添加到MoveIt规划场景。"
            response.success = True; response.message = msg; self.get_logger().info(msg)

        except Exception as e:
            self.get_logger().error(f"处理过程中发生错误: {e}")
            import traceback
            self.get_logger().error(f"错误追溯: {traceback.format_exc()}")
            response.success = False; response.message = str(e)
        return response

    # --- 新增: 用于与MoveIt交互的函数 ---
    def add_mesh_to_planning_scene(self, name: str, o3d_mesh: o3d.geometry.TriangleMesh):
        """将Open3D网格作为CollisionObject添加到MoveIt规划场景"""
        if self.scene is None: return

        co = CollisionObject()
        co.id = name
        # 障碍物是在相机坐标系下检测的，所以frame_id是camera_frame
        # MoveIt会自动处理从camera_frame到base_link的TF转换
        co.header.frame_id = self.camera_frame
        co.operation = CollisionObject.ADD

        mesh = Mesh()
        vertices = np.asarray(o3d_mesh.vertices)
        for v in vertices:
            p = Point(); p.x, p.y, p.z = v; mesh.vertices.append(p)
        
        triangles = np.asarray(o3d_mesh.triangles)
        for t in triangles:
            mt = MeshTriangle(); mt.vertex_indices = [t[0], t[1], t[2]]; mesh.triangles.append(mt)
        
        co.meshes.append(mesh)

        # 定义一个空的姿态，因为网格顶点已经在正确的坐标系下了
        mesh_pose = Pose(); mesh_pose.orientation.w = 1.0
        co.mesh_poses.append(mesh_pose)
        
        self.scene.add_object(co)
        self.get_logger().info(f"  -> 已将 '{name}' 添加到规划场景。")
        time.sleep(0.1)

    def clear_all_obstacles(self):
        """从规划场景中移除所有之前添加的障碍物并清除RViz标记"""
        if self.scene is None: return
        
        object_names = self.scene.get_known_object_names()
        for name in object_names:
            if name.startswith("obstacle_"):
                self.scene.remove_world_object(name)
        self.get_logger().info("已从MoveIt规划场景中清除所有旧障碍物。")
        self.clear_markers()

    # create_mesh_marker函数 (与您的版本相同，只是改了颜色)
    def create_mesh_marker(self, mesh: o3d.geometry.TriangleMesh, marker_id: int) -> Marker:
        marker = Marker(); marker.header.frame_id = self.camera_frame; marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "obstacle_meshes"; marker.id = marker_id; marker.type = Marker.TRIANGLE_LIST; marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 1.0; marker.scale.y = 1.0; marker.scale.z = 1.0
        marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.3) # 您的版本是绿色，我保留了
        vertices = np.asarray(mesh.vertices); triangles = np.asarray(mesh.triangles)
        for triangle_indices in triangles:
            for index in triangle_indices:
                p = Point(); p.x, p.y, p.z = vertices[index]; marker.points.append(p)
        return marker

    # clear_markers函数 (与您的版本相同)
    def clear_markers(self):
        marker_array = MarkerArray()
        clear_marker = Marker(); clear_marker.header.frame_id = self.camera_frame; clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)
        self.marker_pub.publish(marker_array)
        self.get_logger().info("已清除所有障碍物标记。")

def main(args=None):
    rclpy.init(args=args)
    # --- 新增: 为MoveIt Commander初始化ROS参数 ---
    moveit_commander.roscpp_initialize(sys.argv)
    
    node = None
    try:
        node = ObstacleGeometryNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node:
            node.destroy_node()
        # --- 新增: 关闭MoveIt Commander ---
        moveit_commander.roscpp_shutdown()
        rclpy.shutdown()

if __name__ == '__main__':
    main()