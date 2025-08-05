#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import open3d as o3d
import cv2  # 确保导入了 OpenCV
import time

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from moveit_commander import PlanningSceneInterface
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import Mesh, MeshTriangle
from geometry_msgs.msg import Pose, Point

# 导入您自己的分割和点云工具
# 确保这个路径是正确的，并且cv_segmentation.py已经更新
import sys
sys.path.append('/home/roar/graspnet/graspnet-baseline/kinova_graspnet_ros2/utils')
from cv_segmentation import SmartSegmentation

# 导入新的服务定义
from kinova_graspnet_ros2.srv import BuildObstacles

# 【修改1】从 grasp_detection_service.py 导入所需的辅助函数
# 为了保持代码独立和清晰，我们直接导入，而不是依赖于运行grasp_detection_service
# 假设 grasp_detection_service.py 与此文件在同一Python包路径下
# 如果不在，需要调整sys.path
try:
    from grasp_detection_service import create_point_cloud_from_depth_image, GraspNetCameraInfo
except ImportError:
    # 如果直接导入失败，提供一个后备定义，避免程序崩溃
    print("警告：无法从grasp_detection_service导入，将使用本地定义。")
    class GraspNetCameraInfo:
        def __init__(self, width, height, fx, fy, cx, cy, scale):
            self.width = width
            self.height = height
            self.fx = fx
            self.fy = fy
            self.cx = cx
            self.cy = cy
            self.scale = scale

    def create_point_cloud_from_depth_image(depth, camera, organized=True):
        assert(depth.shape[0] == camera.height and depth.shape[1] == camera.width)
        xmap = np.arange(camera.width)
        ymap = np.arange(camera.height)
        xmap, ymap = np.meshgrid(xmap, ymap)
        points_z = depth / camera.scale
        points_x = (xmap - camera.cx) * points_z / camera.fx
        points_y = (ymap - camera.cy) * points_z / camera.fy
        points = np.stack([points_x, points_y, points_z], axis=-1)
        if not organized:
            points = points.reshape([-1, 3])
        return points


class ObstacleGeometryNode(Node):
    """
    一个ROS2服务节点，用于检测场景中的障碍物，
    并将其作为凸包（Convex Hull）碰撞体添加到MoveIt规划场景中。
    """
    def __init__(self):
        super().__init__('obstacle_geometry_node')

        # 【修改2】声明与 grasp_detection_service.py 完全一致的相机参数
        # 这样可以保证点云生成逻辑的一致性
        self.declare_parameter('color_image_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_image_topic', '/camera/depth/image_raw')
        self.declare_parameter('depth_camera_frame', 'camera_depth_frame')
        
        # 使用深度相机的内参，因为点云是基于深度图生成的
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
        # 【修改3】初始化我们自己的分割器实例
        self.segmentation_module = SmartSegmentation(device='cuda:0') # 可以指定设备

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
            self.get_logger().info(f"Removing {len(self.obstacle_names)} old obstacles from planning scene...")
            self.planning_scene_interface.remove_collision_objects(self.obstacle_names)
            self.obstacle_names.clear()
            time.sleep(0.5) # 等待场景更新

        # 3. 数据转换和分割
        try:
            # ROS Image -> OpenCV BGR
            color_image_cv = self.bridge.imgmsg_to_cv2(self.latest_color_image, "bgr8")
            depth_image_cv = self.bridge.imgmsg_to_cv2(self.latest_depth_image, "passthrough")

            # 【修改4】调用我们新增的全局分割方法
            # 这是最核心的修改！
            self.get_logger().info("Calling segment_all_objects...")
            segmented_objects = self.segmentation_module.segment_all_objects(color_image_cv)

            if not segmented_objects:
                response.success = True
                response.message = "No objects detected by segmentation module."
                self.get_logger().info(response.message)
                return response

        except Exception as e:
            response.success = False
            response.message = f"Failed during image conversion or segmentation: {e}"
            self.get_logger().error(response.message, exc_info=True) # 打印详细错误
            return response

        # 4. 创建完整点云
        h, w = depth_image_cv.shape
        camera = GraspNetCameraInfo(w, h, self.depth_cam_fx, self.depth_cam_fy, self.depth_cam_cx, self.depth_cam_cy, 1000.0)
        # 【修改5】创建有组织点云，方便索引
        full_cloud_np = create_point_cloud_from_depth_image(depth_image_cv, camera, organized=True)

        # 5. 迭代处理每个障碍物
        obstacle_count = 0
        for class_name, masks in segmented_objects.items():
            # 【修改6】精确排除目标物体
            if class_name == request.target_object_class:
                self.get_logger().info(f'Skipping target object: "{class_name}"')
                continue

            self.get_logger().info(f'Processing obstacle: "{class_name}"')
            for i, mask in enumerate(masks):
                # 调整掩码尺寸以匹配深度图
                if mask.shape[:2] != (h, w):
                    # 使用OpenCV进行缩放
                    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                else:
                    mask_resized = mask
                
                # 提取点云簇
                # 使用布尔索引，确保掩码和点云形状匹配
                obstacle_points = full_cloud_np[mask_resized > 0]
                
                if len(obstacle_points) < 100: # 忽略太小的点云簇
                    self.get_logger().warn(f'Skipping small point cloud cluster for {class_name}_{i} ({len(obstacle_points)} points).')
                    continue

                obstacle_pcd = o3d.geometry.PointCloud()
                obstacle_pcd.points = o3d.utility.Vector3dVector(obstacle_points)
                
                # 【修改7】增加一步降采样，可以使凸包计算更稳定且更快
                obstacle_pcd = obstacle_pcd.voxel_down_sample(voxel_size=0.005)

                # 6. 计算凸包
                try:
                    hull_mesh, _ = obstacle_pcd.compute_convex_hull()
                    if not hull_mesh.has_triangles():
                        self.get_logger().warn(f"Convex hull for {class_name}_{i} resulted in an empty mesh.")
                        continue
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
            ros_mesh.vertices.append(Point(x=float(v[0]), y=float(v[1]), z=float(v[2])))
            
        triangles = np.asarray(mesh_o3d.triangles)
        for t in triangles:
            ros_mesh.triangles.append(MeshTriangle(vertex_indices=t.astype(np.uint32).tolist()))
            
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
        # 【修改8】使用多线程执行器，避免服务回调阻塞主循环或订阅
        executor = rclpy.executors.MultiThreadedExecutor()
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()