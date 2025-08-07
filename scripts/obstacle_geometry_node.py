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
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

# --- 核心修改: 不再需要moveit_commander, 而是导入消息类型 ---
from moveit_msgs.msg import PlanningScene, CollisionObject
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
    一个ROS2节点，识别障碍物，生成Alpha Shape，
    并直接发布到 /planning_scene 话题以更新MoveIt。
    """
    def __init__(self):
        super().__init__('obstacle_geometry_node')

        # 声明参数 (与您的版本相同)
        self.declare_parameter('color_image_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_image_topic', '/camera/depth_registered/image_rect')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('depth_camera_frame', 'camera_depth_frame')

        # 获取参数
        color_topic = self.get_parameter('color_image_topic').value
        depth_topic = self.get_parameter('depth_image_topic').value
        info_topic = self.get_parameter('camera_info_topic').value
        self.camera_frame = self.get_parameter('depth_camera_frame').value

        # 初始化工具 (与您的版本相同)
        self.bridge = CvBridge()
        self.latest_color_image: Optional[Image] = None
        self.latest_depth_image: Optional[Image] = None
        self.latest_camera_info: Optional[CameraInfo] = None
        self.camera_data_ready = False


        # 使用QoS配置来确保消息传递的可靠性 ---
        sensor_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.VOLATILE
        )
        
        # --- 新增: 用于追踪已添加障碍物的列表 ---
        self.obstacle_names = []

        # 订阅和发布
        self.color_sub = self.create_subscription(
            Image, color_topic, self.color_callback, sensor_qos_profile
        )
        self.depth_sub = self.create_subscription(
            Image, depth_topic, self.depth_callback, sensor_qos_profile
        )
        self.info_sub = self.create_subscription(
            CameraInfo, info_topic, self.info_callback, sensor_qos_profile
        )

        self.marker_pub = self.create_publisher(MarkerArray, 'obstacle_markers', 10)
        
        # --- 核心修改: 创建一个PlanningScene的发布者 ---
        self.planning_scene_pub = self.create_publisher(PlanningScene, '/planning_scene', 10)

        self.obstacle_service = self.create_service(
            GenerateObstacles,
            'generate_obstacles',
            self.generate_obstacles_callback
        )

        self.get_logger().info('Obstacle Geometry Node (Direct Publisher) 已初始化...')

    # 相机回调函数 (与您的版本相同)
    def color_callback(self, msg: Image): self.latest_color_image = msg; self.check_data_ready()
    def depth_callback(self, msg: Image): self.latest_depth_image = msg; self.check_data_ready()
    def info_callback(self, msg: CameraInfo): self.latest_camera_info = msg; self.check_data_ready()
    def check_data_ready(self):
        if not self.camera_data_ready:
            if self.latest_color_image and self.latest_depth_image and self.latest_camera_info:
                self.camera_data_ready = True; self.get_logger().info('相机数据已准备就绪。')

    def generate_obstacles_callback(self, request: GenerateObstacles.Request, response: GenerateObstacles.Response):
        if not self.camera_data_ready:
            response.success = False; response.message = "相机数据尚未准备好。"; self.get_logger().warn(response.message)
            return response
            
        self.get_logger().info(f"收到障碍物生成请求，目标物体为: '{request.target_object_class}'")
        self.clear_all_obstacles()

        try:
            # 步骤 1: 获取图像并生成初始点云
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

            # 步骤 2: 分割目标物体并膨胀掩码
            self.get_logger().info(f"步骤 2/5: 分割目标物体 '{request.target_object_class}'...")
            target_mask, _ = segment_objects(color_cv, target_class=request.target_object_class, return_vis=True)
            if target_mask is None: target_mask = np.zeros(color_cv.shape[:2], dtype=np.uint8)
            kernel = np.ones((13, 13), np.uint8)
            dilated_target_mask = cv2.dilate(target_mask, kernel, iterations=1)
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
                distances = workspace_pcd.compute_point_cloud_distance(target_pcd)
                obstacle_indices = np.where(np.asarray(distances) > 0.005)[0]
                obstacle_pcd = workspace_pcd.select_by_index(obstacle_indices)
            else:
                obstacle_pcd = workspace_pcd
            self.get_logger().info(f"  - 计算后的障碍物点数: {len(obstacle_pcd.points)}")

            pcd = obstacle_pcd
            if len(pcd.points) < 20: 
                self.get_logger().info("过滤后无障碍物需要处理。"); response.success = True; response.message = "无障碍物。"; return response

            pcd = pcd.voxel_down_sample(0.01)
            self.get_logger().info(f"  - 体素下采样后点数: {len(pcd.points)}")
            
            # 步骤 4: 聚类
            self.get_logger().info(f"步骤 4/5: 对 {len(pcd.points)} 个障碍物点进行聚类...")
            labels = np.array(pcd.cluster_dbscan(eps=0.025, min_points=30, print_progress=False))
            max_label = labels.max()
            self.get_logger().info(f"聚类完成，发现 {max_label + 1} 个初始聚类。")
            # --- 补全结束 ---
            
            # 步骤 5: 生成、拆分并发布网格
            self.get_logger().info("步骤 5/5: 生成、拆分网格并更新规划场景...")
            marker_array = MarkerArray()
            final_collision_objects = []
            
            alpha_value = 0.035
            
            for i in range(max_label + 1):
                cluster_indices = np.where(labels == i)[0]
                if len(cluster_indices) < 4: continue
                
                cluster_pcd = pcd.select_by_index(cluster_indices)
                try:
                    initial_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(cluster_pcd, alpha_value)
                    initial_mesh.compute_vertex_normals()

                    triangle_clusters, cluster_n_triangles, _ = initial_mesh.cluster_connected_triangles()
                    triangle_clusters = np.asarray(triangle_clusters)
                    
                    max_component_label = triangle_clusters.max()
                    for comp_idx in range(max_component_label + 1):
                        if cluster_n_triangles[comp_idx] < 10: continue

                        component_mesh = o3d.geometry.TriangleMesh()
                        component_mesh.vertices = initial_mesh.vertices
                        component_triangles = np.asarray(initial_mesh.triangles)[triangle_clusters == comp_idx]
                        component_mesh.triangles = o3d.utility.Vector3iVector(component_triangles)
                        component_mesh.remove_unreferenced_vertices()
                        component_mesh.orient_triangles()
                        
                        obstacle_id = len(final_collision_objects)
                        obstacle_name = f"obstacle_{obstacle_id}"
                        
                        co = self.create_collision_object_from_mesh(obstacle_name, component_mesh)
                        final_collision_objects.append(co)
                        self.obstacle_names.append(obstacle_name)
                        
                        mesh_marker = self.create_mesh_marker(component_mesh, obstacle_id)
                        marker_array.markers.append(mesh_marker)
                        
                except Exception as e:
                    self.get_logger().warn(f"为聚类 {i} 处理网格时发生错误: {e}")

            if final_collision_objects:
                planning_scene_msg = PlanningScene()
                planning_scene_msg.is_diff = True
                planning_scene_msg.world.collision_objects = final_collision_objects
                self.planning_scene_pub.publish(planning_scene_msg)
                self.get_logger().info(f"已将 {len(final_collision_objects)} 个【独立】障碍物发布到规划场景。")

            self.marker_pub.publish(marker_array)
            
            msg = f"成功生成并发布了 {len(final_collision_objects)} 个【独立】障碍物。"
            response.success = True; response.message = msg; self.get_logger().info(msg)

        except Exception as e:
            self.get_logger().error(f"生成障碍物时发生错误: {e}")
            import traceback
            self.get_logger().error(f"错误追溯: {traceback.format_exc()}")
            response.success = False; response.message = f"生成障碍物失败: {e}"
        return response


    def create_collision_object_from_mesh(self, name: str, o3d_mesh: o3d.geometry.TriangleMesh) -> CollisionObject:
        """从Open3D网格创建一个CollisionObject消息"""
        co = CollisionObject(); co.id = name; co.header.frame_id = self.camera_frame
        co.operation = CollisionObject.ADD
        mesh = Mesh()
        vertices = np.asarray(o3d_mesh.vertices)
        for v in vertices:
            p = Point(); p.x, p.y, p.z = v; mesh.vertices.append(p)
        triangles = np.asarray(o3d_mesh.triangles)
        for t in triangles:
            mt = MeshTriangle()
            mt.vertex_indices = [int(t[0]), int(t[1]), int(t[2])]
            mesh.triangles.append(mt)
        co.meshes.append(mesh)
        mesh_pose = Pose(); mesh_pose.orientation.w = 1.0; co.mesh_poses.append(mesh_pose)
        return co

    def clear_all_obstacles(self):
        """发布一个移除所有旧障碍物的PlanningScene消息"""
        if not self.obstacle_names: return
        
        collision_objects_to_remove = []
        for name in self.obstacle_names:
            co = CollisionObject(); co.id = name; co.operation = CollisionObject.REMOVE
            collision_objects_to_remove.append(co)
            
        planning_scene_msg = PlanningScene()
        planning_scene_msg.is_diff = True
        planning_scene_msg.world.collision_objects = collision_objects_to_remove
        self.planning_scene_pub.publish(planning_scene_msg)
        
        self.get_logger().info(f"已从MoveIt规划场景中清除 {len(self.obstacle_names)} 个旧障碍物。")
        self.obstacle_names = [] # 清空列表
        self.clear_markers()

    def create_mesh_marker(self, mesh: o3d.geometry.TriangleMesh, marker_id: int) -> Marker:
        # (与您的版本相同)
        marker = Marker(); marker.header.frame_id = self.camera_frame; marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "obstacle_meshes"; marker.id = marker_id; marker.type = Marker.TRIANGLE_LIST; marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0; marker.scale.x = 1.0; marker.scale.y = 1.0; marker.scale.z = 1.0
        marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.3)
        vertices = np.asarray(mesh.vertices); triangles = np.asarray(mesh.triangles)
        for triangle_indices in triangles:
            for index in triangle_indices:
                p = Point(); p.x, p.y, p.z = vertices[index]; marker.points.append(p)
        return marker

    def clear_markers(self):
        # (与您的版本相同)
        marker_array = MarkerArray(); clear_marker = Marker(); clear_marker.header.frame_id = self.camera_frame; clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker); self.marker_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    # --- 核心修改: 不再需要moveit_commander的初始化和关闭 ---
    node = None
    try:
        node = ObstacleGeometryNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node:
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()