#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import open3d as o3d
import cv2
import time
from scipy.spatial.transform import Rotation as R

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from moveit_msgs.msg import CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene
from shape_msgs.msg import Mesh, MeshTriangle
from geometry_msgs.msg import Pose, Point, Quaternion

import tf2_ros
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

# 【修复2】导入traceback模块用于打印详细错误
import traceback

import sys
sys.path.append('/home/roar/graspnet/graspnet-baseline/kinova_graspnet_ros2/utils')
from cv_segmentation import SmartSegmentation
from kinova_graspnet_ros2.srv import BuildObstacles

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

        self.declare_parameter('color_image_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_image_topic', '/camera/depth/image_raw')
        self.declare_parameter('depth_camera_frame', 'camera_link')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('depth_camera_intrinsics.fx', 336.190430)
        self.declare_parameter('depth_camera_intrinsics.fy', 336.190430)
        self.declare_parameter('depth_camera_intrinsics.cx', 230.193512)
        self.declare_parameter('depth_camera_intrinsics.cy', 138.111130)

        self.color_topic = self.get_parameter('color_image_topic').value
        self.depth_topic = self.get_parameter('depth_image_topic').value
        self.camera_frame = self.get_parameter('depth_camera_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.depth_cam_fx = self.get_parameter('depth_camera_intrinsics.fx').value
        self.depth_cam_fy = self.get_parameter('depth_camera_intrinsics.fy').value
        self.depth_cam_cx = self.get_parameter('depth_camera_intrinsics.cx').value
        self.depth_cam_cy = self.get_parameter('depth_camera_intrinsics.cy').value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.bridge = CvBridge()
        self.segmentation_module = SmartSegmentation(device='cuda:0')
        self.apply_planning_scene_client = self.create_client(ApplyPlanningScene, '/apply_planning_scene')
        while not self.apply_planning_scene_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info('Waiting for /apply_planning_scene service...')

        self.latest_color_image, self.latest_depth_image = None, None
        self.color_sub = self.create_subscription(Image, self.color_topic, self.color_image_callback, 10)
        self.depth_sub = self.create_subscription(Image, self.depth_topic, self.depth_image_callback, 10)
        self.obstacle_names = []
        self.build_obstacles_srv = self.create_service(BuildObstacles, 'build_obstacles', self.build_obstacles_callback)
        self.get_logger().info('Obstacle Geometry Node is ready (Client-side Transform, Final Fix).')

    def color_image_callback(self, msg: Image): self.latest_color_image = msg
    def depth_image_callback(self, msg: Image): self.latest_depth_image = msg

    def build_obstacles_callback(self, request: BuildObstacles.Request, response: BuildObstacles.Response):
        self.get_logger().info(f'Received request to build obstacles, excluding "{request.target_object_class}".')
        if self.latest_color_image is None or self.latest_depth_image is None:
            response.success, response.message = False, "Camera images not available."
            return response
        
        self.clear_obstacles()
        
        try:
            color_image_cv = self.bridge.imgmsg_to_cv2(self.latest_color_image, "bgr8")
            depth_image_cv = self.bridge.imgmsg_to_cv2(self.latest_depth_image, "passthrough")
            segmented_objects = self.segmentation_module.segment_all_objects(color_image_cv)
        except Exception as e:
            response.success, response.message = False, f"Segmentation failed: {e}"
            self.get_logger().error(f"{response.message}\n{traceback.format_exc()}")
            return response

        if not segmented_objects:
            response.success, response.message = True, "No objects detected."
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
                try:
                    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST) if mask.shape[:2] != (h, w) else mask
                    obstacle_points = full_cloud_np[mask_resized > 0]
                    if len(obstacle_points) < 100: continue

                    obstacle_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(obstacle_points)).voxel_down_sample(0.005)
                    hull_mesh, _ = obstacle_pcd.compute_convex_hull()
                    if not hull_mesh.has_triangles(): continue

                    transformed_mesh = self.transform_mesh(hull_mesh, self.camera_frame, self.base_frame)
                    if transformed_mesh is None:
                        self.get_logger().warn(f"Failed to transform mesh for {class_name}_{i}, skipping.")
                        continue
                    
                    obstacle_name = f"obstacle_{class_name}_{i}"
                    self.add_mesh_to_planning_scene_service(obstacle_name, transformed_mesh, self.base_frame)
                    self.obstacle_names.append(obstacle_name)
                    obstacle_count += 1
                except Exception as e:
                    # 【修复2】修正日志打印方式
                    self.get_logger().error(f"Error processing {class_name}_{i}: {e}\n{traceback.format_exc()}")

        response.success, response.message, response.num_obstacles_built = True, f"Successfully built {obstacle_count} obstacles.", obstacle_count
        return response

    def transform_mesh(self, mesh_in, source_frame, target_frame):
        try:
            trans = self.tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=2.0))
            t, q = trans.transform.translation, trans.transform.rotation
            mat = np.eye(4)
            mat[:3, :3] = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
            mat[:3, 3] = [t.x, t.y, t.z]
            return o3d.geometry.TriangleMesh(mesh_in).transform(mat)
        except TransformException as ex:
            self.get_logger().error(f'Could not transform {source_frame} to {target_frame}: {ex}')
            return None

    def add_mesh_to_planning_scene_service(self, name: str, mesh_o3d: o3d.geometry.TriangleMesh, frame_id: str):
        # 【修复1】修正CollisionObject的创建方式
        co = CollisionObject()
        co.header.stamp = self.get_clock().now().to_msg() # 使用节点时钟
        co.header.frame_id = frame_id
        co.id = name
        co.operation = CollisionObject.ADD

        ros_mesh = Mesh()
        ros_mesh.vertices = [Point(x=v[0], y=v[1], z=v[2]) for v in np.asarray(mesh_o3d.vertices)]
        ros_mesh.triangles = [MeshTriangle(vertex_indices=t.astype(np.uint32).tolist()) for t in np.asarray(mesh_o3d.triangles)]
        
        co.meshes.append(ros_mesh)
        co.mesh_poses.append(Pose(orientation=Quaternion(w=1.0)))

        request = ApplyPlanningScene.Request()
        scene = PlanningScene()
        scene.world.collision_objects.append(co)
        scene.is_diff = True
        request.scene = scene
        
        future = self.apply_planning_scene_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

        if future.result():
            if future.result().success:
                self.get_logger().info(f"Added collision object '{name}' to scene in '{frame_id}'.")
            else:
                self.get_logger().error(f"Failed to add '{name}'. Service returned failure.")
        else:
            # 这里的超时错误现在是预期的，因为我们还在调试
            self.get_logger().error(f"Service call to add '{name}' timed out. Check move_group logs.")

    def clear_obstacles(self):
        if not self.obstacle_names: return
        self.get_logger().info(f"Removing {len(self.obstacle_names)} old obstacles...")
        scene = PlanningScene(is_diff=True)
        for name in self.obstacle_names:
            co = CollisionObject()
            co.id = name
            co.operation = CollisionObject.REMOVE
            scene.world.collision_objects.append(co)
        
        request = ApplyPlanningScene.Request(scene=scene)
        future = self.apply_planning_scene_client.call_async(request)
        # 清理时可以设置一个短一点的超时
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        self.obstacle_names.clear()

def main(args=None):
    rclpy.init(args=args)
    try:
        node = ObstacleGeometryNode()
        executor = rclpy.executors.MultiThreadedExecutor()
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt: pass
    finally:
        # 在关闭前销毁节点
        if 'node' in locals() and rclpy.ok():
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()