#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import open3d as o3d
import time

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from moveit_commander import PlanningSceneInterface
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import Mesh, MeshTriangle
from geometry_msgs.msg import Pose, Point

# 导入您自己的分割和点云工具
# 确保这个路径是正确的
import sys
sys.path.append('/home/roar/graspnet/graspnet-baseline/kinova_graspnet_ros2/utils')
from cv_segmentation import SmartSegmentation

# 导入新的服务定义
from kinova_graspnet_ros2.srv import BuildObstacles

# 从 grasp_detection_service.py 借用点云创建函数，以保证一致性
from grasp_detection_service import create_point_cloud_from_depth_image, GraspNetCameraInfo


class ObstacleGeometryNode(Node):
    """
    一个ROS2服务节点，用于检测场景中的障碍物，
    并将其作为凸包（Convex Hull）碰撞体添加到MoveIt规划场景中。
    """
    def __init__(self):
        super().__init__('obstacle_geometry_node')

        # 声明与其它节点一致的参数
        self.declare_parameter('color_image_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_image_topic', '/camera/depth/image_raw')
        self.declare_parameter('depth_camera_frame', 'camera_depth_frame')
        self.declare_parameter('depth_camera_intrinsics.fx', 336.190430)
        self.declare_parameter('depth_camera_intrinsics.fy', 336.190430)
        self.declare_parameter('depth_camera_intrinsics.cx', 230.193512)
        self.declare_parameter('depth_camera_intrinsics.cy', 138.111130)

        # 获取参数
        self.color_topic = self.get_parameter('color_image_topic').value
        self.depth_topic = self.get_parameter('depth_image_topic').value
        self.camera_frame = self.get_parameter('depth_camera_frame').value
        self.depth_cam_fx = self.get_parameter('depth_camera_intrinsics.fx').value
        self.depth_cam_fy = self.get_parameter('depth_camera_intrinsics.fy').value
        self.depth_cam_cx = self.get_parameter('depth_camera_intrinsics.cx').value
        self.depth_cam_cy = self.get_parameter('depth_camera_intrinsics.cy').value

        # 初始化工具
        self.bridge = CvBridge()
        self.planning_scene_interface = PlanningSceneInterface(synchronous=True)
        self.segmentation_module = SmartSegmentation()

        # 订阅相机话题
        self.latest_color_image = None
        self.latest_depth_image = None
        self.color_sub = self.create_subscription(Image, self.color_topic, self.color_image_callback, 10)
        self.depth_sub = self.create_subscription(Image, self.depth_topic, self.depth_image_callback, 10)
        
        # 记录已添加的障碍物名称，方便后续清理
        self.obstacle_names = []

        # 创建服务
        self.build_obstacles_srv = self.create_service(
            BuildObstacles,
            'build_obstacles',
            self.build_obstacles_callback
        )

        self.get_logger().info('Obstacle Geometry Node is ready.')
        self.get_logger().info('Call "ros2 service call /build_obstacles kinova_graspnet_ros2/srv/BuildObstacles \'{target_object_class: "bottle"}\'" to build obstacles.')

    def color_image_callback(self, msg: Image):
        self.latest_color_image = msg

    def depth_image_callback(self, msg: Image):
        self.latest_depth_image = msg

    def build_obstacles_callback(self, request: BuildObstacles.Request, response: BuildObstacles.Response):
        """服务回调函数，执行障碍物建模和发布"""
        self.get_logger().info(f'Received request to build obstacles, excluding "{request.target_object_class}".')

        # 1. 检查数据是否就绪
        if self.latest_color_image is None or self.latest_depth_image is None:
            response.success = False
            response.message = "Camera images are not available."
            self.get_logger().error(response.message)
            return response
            
        # 2. 清理旧的障碍物
        if self.obstacle_names:
            self.get_logger().info(f"Removing {len(self.obstacle_names)} old obstacles...")
            for name in self.obstacle_names:
                self.planning_scene_interface.remove_collision_object(name)
            self.obstacle_names.clear()
            time.sleep(0.5) # 等待场景更新

        # 3. 数据转换和分割
        try:
            color_image_cv = self.bridge.imgmsg_to_cv2(self.latest_color_image, "rgb8")
            depth_image_cv = self.bridge.imgmsg_to_cv2(self.latest_depth_image, "passthrough")

            # 调用分割模块，获取所有物体的掩码
            # 注意: 这里假设 segment_all_objects 返回一个字典 {class_name: [mask1, mask2...]}
            segmented_objects = self.segmentation_module.segment_all_objects(color_image_cv)

            if not segmented_objects:
                response.success = True
                response.message = "No objects detected by segmentation."
                self.get_logger().info(response.message)
                return response

        except Exception as e:
            response.success = False
            response.message = f"Failed during image conversion or segmentation: {e}"
            self.get_logger().error(response.message)
            return response

        # 4. 创建完整点云
        h, w = depth_image_cv.shape
        camera = GraspNetCameraInfo(w, h, self.depth_cam_fx, self.depth_cam_fy, self.depth_cam_cx, self.depth_cam_cy, 1000.0)
        full_cloud_o3d = create_point_cloud_from_depth_image(depth_image_cv, camera, organized=True)
        full_cloud_np = np.asarray(full_cloud_o3d.points).reshape((h, w, 3))

        # 5. 迭代处理每个障碍物
        obstacle_count = 0
        for class_name, masks in segmented_objects.items():
            if class_name == request.target_object_class:
                self.get_logger().info(f'Skipping target object: "{class_name}"')
                continue

            self.get_logger().info(f'Processing obstacle: "{class_name}"')
            for i, mask in enumerate(masks):
                # 调整掩码尺寸以匹配深度图
                if mask.shape[:2] != (h, w):
                    mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                
                # 提取点云簇
                obstacle_points = full_cloud_np[mask > 0]
                
                if len(obstacle_points) < 100: # 忽略太小的点云簇
                    self.get_logger().warn(f'Skipping small point cloud cluster for {class_name}_{i} ({len(obstacle_points)} points).')
                    continue

                obstacle_pcd = o3d.geometry.PointCloud()
                obstacle_pcd.points = o3d.utility.Vector3dVector(obstacle_points)
                
                # 6. 计算凸包
                try:
                    hull_mesh, _ = obstacle_pcd.compute_convex_hull()
                    hull_mesh.compute_vertex_normals() # 好习惯
                except Exception as e:
                    self.get_logger().error(f"Failed to compute convex hull for {class_name}_{i}: {e}")
                    continue

                # 7. 创建并发布CollisionObject
                obstacle_name = f"obstacle_{class_name}_{i}"
                self.add_mesh_to_planning_scene(obstacle_name, hull_mesh)
                self.obstacle_names.append(obstacle_name)
                obstacle_count += 1
                
        response.success = True
        response.message = f"Successfully built {obstacle_count} obstacles."
        response.num_obstacles_built = obstacle_count
        self.get_logger().info(response.message)
        return response

    def add_mesh_to_planning_scene(self, name: str, mesh_o3d: o3d.geometry.TriangleMesh):
        """将open3d的Mesh作为碰撞体添加到MoveIt规划场景中"""
        
        collision_object = CollisionObject()
        collision_object.header.frame_id = self.camera_frame # 障碍物位于相机坐标系中
        collision_object.id = name
        
        # 从Open3D Mesh创建ROS Mesh消息
        ros_mesh = Mesh()
        vertices = np.asarray(mesh_o3d.vertices)
        for v in vertices:
            ros_mesh.vertices.append(Point(x=v[0], y=v[1], z=v[2]))
            
        triangles = np.asarray(mesh_o3d.triangles)
        for t in triangles:
            ros_mesh.triangles.append(MeshTriangle(vertex_indices=t.astype(np.uint32)))
            
        # 设置碰撞体的形状和位姿
        # 因为点云和Mesh顶点都在相机坐标系下，所以位姿是单位矩阵（无变换）
        mesh_pose = Pose()
        mesh_pose.orientation.w = 1.0 # 单位四元数
        
        collision_object.meshes.append(ros_mesh)
        collision_object.mesh_poses.append(mesh_pose)
        collision_object.operation = CollisionObject.ADD

        # 应用到规划场景
        self.planning_scene_interface.apply_collision_object(collision_object)
        self.get_logger().info(f"Added collision object '{name}' to planning scene.")


def main(args=None):
    rclpy.init(args=args)
    try:
        node = ObstacleGeometryNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()