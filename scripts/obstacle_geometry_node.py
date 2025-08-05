#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import open3d as o3d
import cv2
import time

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# 【修改1】导入新的、更底层的MoveIt消息和服务类型
from moveit_msgs.msg import CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene
from shape_msgs.msg import Mesh, MeshTriangle
from geometry_msgs.msg import Pose, Point

# 导入您自己的分割和点云工具
import sys
sys.path.append('/home/roar/graspnet/graspnet-baseline/kinova_graspnet_ros2/utils')
from cv_segmentation import SmartSegmentation
from kinova_graspnet_ros2.srv import BuildObstacles

# 导入辅助函数
try:
    from grasp_detection_service import create_point_cloud_from_depth_image, GraspNetCameraInfo
except ImportError:
    print("警告：无法从grasp_detection_service导入，将使用本地定义。")
    class GraspNetCameraInfo:
        def __init__(self, width, height, fx, fy, cx, cy, scale):
            self.width, self.height, self.fx, self.fy, self.cx, self.cy, self.scale = width, height, fx, fy, cx, cy, scale

    def create_point_cloud_from_depth_image(depth, camera, organized=True):
        xmap, ymap = np.meshgrid(np.arange(camera.width), np.arange(camera.height))
        points_z = depth / camera.scale
        points_x = (xmap - camera.cx) * points_z / camera.fx
        points_y = (ymap - camera.cy) * points_z / camera.fy
        points = np.stack([points_x, points_y, points_z], axis=-1)
        return points if organized else points.reshape([-1, 3])

class ObstacleGeometryNode(Node):
    def __init__(self):
        super().__init__('obstacle_geometry_node')

        # 参数声明 (与之前保持一致)
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
        self.segmentation_module = SmartSegmentation(device='cuda:0')

        # 【修改2】移除 PlanningSceneInterface，创建服务客户端
        # del self.planning_scene_interface
        self.apply_planning_scene_client = self.create_client(
            ApplyPlanningScene,
            '/apply_planning_scene'
        )
        while not self.apply_planning_scene_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info('Waiting for /apply_planning_scene service...')

        # 订阅相机话题
        self.latest_color_image = None
        self.latest_depth_image = None
        self.color_sub = self.create_subscription(Image, self.color_topic, self.color_image_callback, 10)
        self.depth_sub = self.create_subscription(Image, self.depth_topic, self.depth_image_callback, 10)
        
        self.obstacle_names = []

        # 创建服务
        self.build_obstacles_srv = self.create_service(
            BuildObstacles,
            'build_obstacles',
            self.build_obstacles_callback
        )

        self.get_logger().info('Obstacle Geometry Node is ready (using direct service call).')

    def color_image_callback(self, msg: Image):
        self.latest_color_image = msg

    def depth_image_callback(self, msg: Image):
        self.latest_depth_image = msg

    def build_obstacles_callback(self, request: BuildObstacles.Request, response: BuildObstacles.Response):
        self.get_logger().info(f'Received request to build obstacles, excluding "{request.target_object_class}".')

        if self.latest_color_image is None or self.latest_depth_image is None:
            response.success = False
            response.message = "Camera images are not available."
            self.get_logger().error(response.message)
            return response
            
        # 【修改3】清理旧障碍物的逻辑也需要改变
        self.clear_obstacles()
        
        try:
            color_image_cv = self.bridge.imgmsg_to_cv2(self.latest_color_image, "bgr8")
            depth_image_cv = self.bridge.imgmsg_to_cv2(self.latest_depth_image, "passthrough")
            segmented_objects = self.segmentation_module.segment_all_objects(color_image_cv)

            if not segmented_objects:
                response.success = True
                response.message = "No objects detected."
                return response
        except Exception as e:
            response.success = False
            response.message = f"Segmentation failed: {e}"
            self.get_logger().error(response.message, exc_info=True)
            return response

        h, w = depth_image_cv.shape
        camera = GraspNetCameraInfo(w, h, self.depth_cam_fx, self.depth_cam_fy, self.depth_cam_cx, self.depth_cam_cy, 1000.0)
        full_cloud_np = create_point_cloud_from_depth_image(depth_image_cv, camera, organized=True)

        obstacle_count = 0
        for class_name, masks in segmented_objects.items():
            if class_name == request.target_object_class:
                self.get_logger().info(f'Skipping target object: "{class_name}"')
                continue

            for i, mask in enumerate(masks):
                # ... [此部分逻辑与之前版本相同，直到调用 add_mesh_to_planning_scene] ...
                if mask.shape[:2] != (h, w):
                    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                else:
                    mask_resized = mask
                
                obstacle_points = full_cloud_np[mask_resized > 0]
                
                if len(obstacle_points) < 100:
                    continue

                obstacle_pcd = o3d.geometry.PointCloud()
                obstacle_pcd.points = o3d.utility.Vector3dVector(obstacle_points)
                obstacle_pcd = obstacle_pcd.voxel_down_sample(voxel_size=0.005)

                try:
                    hull_mesh, _ = obstacle_pcd.compute_convex_hull()
                    if not hull_mesh.has_triangles():
                        continue
                    hull_mesh.compute_vertex_normals()
                except Exception as e:
                    self.get_logger().error(f"Convex hull failed for {class_name}_{i}: {e}")
                    continue

                obstacle_name = f"obstacle_{class_name}_{i}"
                # 【修改4】调用我们新的、基于服务的添加函数
                self.add_mesh_to_planning_scene_service(obstacle_name, hull_mesh)
                self.obstacle_names.append(obstacle_name)
                obstacle_count += 1
                
        response.success = True
        response.message = f"Successfully built {obstacle_count} obstacles."
        response.num_obstacles_built = obstacle_count
        return response

    # 【修改5】新增一个专门用于清理障碍物的函数
    def clear_obstacles(self):
        """使用服务调用来清理之前添加的所有障碍物"""
        if not self.obstacle_names:
            return

        self.get_logger().info(f"Removing {len(self.obstacle_names)} old obstacles via service call...")
        
        planning_scene = PlanningScene()
        planning_scene.is_diff = True # 表示这是一个更新

        for name in self.obstacle_names:
            co = CollisionObject()
            co.id = name
            co.operation = CollisionObject.REMOVE # 指定操作为移除
            planning_scene.world.collision_objects.append(co)
        
        # 发送请求
        request = ApplyPlanningScene.Request()
        request.scene = planning_scene
        future = self.apply_planning_scene_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

        if future.result() and future.result().success:
            self.get_logger().info("Old obstacles removed successfully.")
        else:
            self.get_logger().warn("Failed to remove old obstacles.")
        
        self.obstacle_names.clear()

    # 【修改6】将 `add_mesh_to_planning_scene` 彻底改造为服务调用方式
    def add_mesh_to_planning_scene_service(self, name: str, mesh_o3d: o3d.geometry.TriangleMesh):
        """将open3d的Mesh作为碰撞体，通过服务调用添加到MoveIt规划场景中"""
        
        # 1. 创建CollisionObject消息 (这部分不变)
        co = CollisionObject()
        co.header.frame_id = self.camera_frame
        co.id = name
        
        ros_mesh = Mesh()
        vertices = np.asarray(mesh_o3d.vertices)
        for v in vertices:
            ros_mesh.vertices.append(Point(x=float(v[0]), y=float(v[1]), z=float(v[2])))
        
        triangles = np.asarray(mesh_o3d.triangles)
        for t in triangles:
            ros_mesh.triangles.append(MeshTriangle(vertex_indices=t.astype(np.uint32).tolist()))
            
        mesh_pose = Pose()
        mesh_pose.orientation.w = 1.0
        
        co.meshes.append(ros_mesh)
        co.mesh_poses.append(mesh_pose)
        co.operation = CollisionObject.ADD

        # 2. 构建PlanningScene消息
        planning_scene = PlanningScene()
        planning_scene.world.collision_objects.append(co)
        planning_scene.is_diff = True  # 关键！表示只应用这个变更

        # 3. 构建并发送服务请求
        request = ApplyPlanningScene.Request()
        request.scene = planning_scene
        
        future = self.apply_planning_scene_client.call_async(request)

        # 4. 等待服务完成 (可以设置为非阻塞，但这里为了简单设为阻塞等待)
        # 在实际应用中，可以处理future的回调函数
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        
        if future.result():
            if future.result().success:
                self.get_logger().info(f"Added collision object '{name}' to planning scene via service.")
            else:
                self.get_logger().warn(f"Service call to add '{name}' failed.")
        else:
            self.get_logger().error(f"Service call for '{name}' timed out.")


def main(args=None):
    rclpy.init(args=args)
    try:
        node = ObstacleGeometryNode()
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