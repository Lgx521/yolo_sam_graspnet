#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from cv_bridge import CvBridge
import numpy as np
import open3d as o3d
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point
import cv2
import sys

# 假设您的 cv_segmentation.py 在这个路径下
sys.path.append('/home/roar/graspnet/graspnet-baseline/kinova_graspnet_ros2/utils')
from cv_segmentation import segment_objects

# 导入我们刚刚创建的服务
from kinova_graspnet_ros2.srv import GenerateObstacles

# 导入与 grasp_detection_service.py 相同的点云创建工具
sys.path.append('/home/roar/graspnet/graspnet-baseline/utils')
from data_utils import CameraInfo as GraspNetCameraInfo, create_point_cloud_from_depth_image


class ObstacleGeometryNode(Node):
    """
    一个ROS2节点，用于识别除目标物体外的障碍物，
    并将其几何形状建模为凸包进行可视化。
    """
    def __init__(self):
        super().__init__('obstacle_geometry_node')

        # 声明与抓取检测服务一致的参数
        self.declare_parameter('color_image_topic', '/camera/color/image_raw')
        # self.declare_parameter('depth_image_topic', '/camera/depth/image_raw')
        self.declare_parameter('depth_image_topic', '/camera/depth_registered/image_rect')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('depth_camera_frame', 'camera_link')

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

    # def generate_obstacles_callback(self, request: GenerateObstacles.Request, response: GenerateObstacles.Response):
    #     """服务回调函数，执行障碍物检测和凸包生成"""
    #     if not self.camera_data_ready:
    #         response.success = False
    #         response.message = "相机数据尚未准备好。"
    #         self.get_logger().warn(response.message)
    #         return response

    #     self.get_logger().info(f"收到障碍物生成请求，目标物体为: '{request.target_object_class}'")

    #     try:
    #         # 1. 将ROS图像消息转换为OpenCV格式
    #         color_cv = self.bridge.imgmsg_to_cv2(self.latest_color_image, "rgb8")
    #         depth_cv = self.bridge.imgmsg_to_cv2(self.latest_depth_image, "passthrough")

    #         # 2. 智能分割以获取掩码
    #         # a. 分割所有可识别的物体
    #         self.get_logger().info("步骤 1/5: 分割场景中的所有物体...")
    #         all_objects_mask, _ = segment_objects(color_cv, target_class=None, return_vis=True)
    #         if all_objects_mask is None: all_objects_mask = np.zeros(color_cv.shape[:2], dtype=np.uint8)
            
    #         # b. 分割目标物体
    #         self.get_logger().info(f"步骤 2/5: 分割目标物体 '{request.target_object_class}'...")
    #         target_object_mask, _ = segment_objects(color_cv, target_class=request.target_object_class, return_vis=True)
    #         if target_object_mask is None: target_object_mask = np.zeros(color_cv.shape[:2], dtype=np.uint8)

    #         # c. 计算障碍物掩码 (所有物体 - 目标物体)
    #         # 确保掩码是布尔类型以进行逻辑运算
    #         obstacle_mask = (all_objects_mask > 0) & ~(target_object_mask > 0)
            
    #         # 3. 生成点云
    #         self.get_logger().info("步骤 3/5: 从深度图生成点云...")
            
    #         # 使用与grasp_detection_service相同的相机参数逻辑
    #         # 这里我们简化处理，假设深度图与彩色图已配准，使用彩色相机的内参
    #         cam_info = self.latest_camera_info
    #         fx, fy = cam_info.k[0], cam_info.k[4]
    #         cx, cy = cam_info.k[2], cam_info.k[5]
    #         scale = 1000.0  # 假设深度单位是毫米

    #         # 调整掩码尺寸以匹配深度图
    #         if obstacle_mask.shape != depth_cv.shape:
    #             obstacle_mask_resized = cv2.resize(obstacle_mask.astype(np.uint8), (depth_cv.shape[1], depth_cv.shape[0]), interpolation=cv2.INTER_NEAREST)
    #         else:
    #             obstacle_mask_resized = obstacle_mask.astype(np.uint8)

    #         graspnet_cam = GraspNetCameraInfo(depth_cv.shape[1], depth_cv.shape[0], fx, fy, cx, cy, scale)
    #         full_cloud = create_point_cloud_from_depth_image(depth_cv, graspnet_cam, organized=True)
            
    #         # 应用障碍物掩码和有效深度掩码
    #         valid_depth_mask = (depth_cv > 0)
    #         final_obstacle_mask = (obstacle_mask_resized > 0) & valid_depth_mask
            
    #         obstacle_points = full_cloud[final_obstacle_mask]
            
    #         if len(obstacle_points) == 0:
    #             self.get_logger().info("未找到属于障碍物的点云。")
    #             self.clear_markers() # 清除旧的标记
    #             response.success = True
    #             response.message = "未找到障碍物。"
    #             return response

    #         # 4. 对障碍物点云进行聚类 (DBSCAN)
    #         self.get_logger().info(f"步骤 4/5: 对 {len(obstacle_points)} 个障碍物点进行聚类...")
    #         pcd = o3d.geometry.PointCloud()
    #         pcd.points = o3d.utility.Vector3dVector(obstacle_points)
            
    #         # eps: 邻域半径; min_points: 成为核心点的最小邻居数
    #         # 这两个参数可能需要根据您的场景进行微调
    #         labels = np.array(pcd.cluster_dbscan(eps=0.04, min_points=20, print_progress=False))
            
    #         max_label = labels.max()
    #         self.get_logger().info(f"聚类完成，发现 {max_label + 1} 个障碍物。")

    #         # 5. 为每个聚类创建凸包并发布Marker
    #         self.get_logger().info("步骤 5/5: 生成凸包并发布可视化标记...")
    #         marker_array = MarkerArray()
            
    #         # 添加一个用于清除旧标记的Marker
    #         clear_marker = Marker()
    #         clear_marker.header.frame_id = self.camera_frame
    #         clear_marker.action = Marker.DELETEALL
    #         marker_array.markers.append(clear_marker)

    #         for i in range(max_label + 1):
    #             # 提取属于当前聚类的点
    #             cluster_indices = np.where(labels == i)[0]
    #             cluster_pcd = pcd.select_by_index(cluster_indices)
                
    #             # 计算凸包
    #             try:
    #                 hull, _ = cluster_pcd.compute_convex_hull()
    #                 hull.orient_triangles() # 确保法线方向一致
                    
    #                 # 将凸包转换为可视化标记
    #                 hull_marker = self.create_hull_marker(hull, i)
    #                 marker_array.markers.append(hull_marker)
    #             except Exception as e:
    #                 self.get_logger().warn(f"为聚类 {i} 创建凸包失败: {e}")

    #         self.marker_pub.publish(marker_array)
    #         response.success = True
    #         response.message = f"成功生成并发布了 {max_label + 1} 个障碍物的凸包。"
    #         self.get_logger().info(response.message)

    #     except Exception as e:
    #         self.get_logger().error(f"处理过程中发生错误: {e}")
    #         import traceback
    #         self.get_logger().error(f"错误追溯: {traceback.format_exc()}")
    #         response.success = False
    #         response.message = str(e)

    #     return response


    def generate_obstacles_callback(self, request: GenerateObstacles.Request, response: GenerateObstacles.Response):
        """服务回调函数，执行障碍物检测和凸包生成"""
        if not self.camera_data_ready:
            # ... (这部分代码不变)
            return response

        self.get_logger().info(f"收到障碍物生成请求，目标物体为: '{request.target_object_class}'")

        try:
            # --- 1. 和 2. 分割掩码 (这部分代码不变) ---
            color_cv = self.bridge.imgmsg_to_cv2(self.latest_color_image, "rgb8")
            self.get_logger().info("步骤 1/5: 分割场景中的所有物体...")
            all_objects_mask, _ = segment_objects(color_cv, target_class=None, return_vis=True)
            if all_objects_mask is None: all_objects_mask = np.zeros(color_cv.shape[:2], dtype=np.uint8)
            
            self.get_logger().info(f"步骤 2/5: 分割目标物体 '{request.target_object_class}'...")
            target_object_mask, _ = segment_objects(color_cv, target_class=request.target_object_class, return_vis=True)
            if target_object_mask is None: target_object_mask = np.zeros(color_cv.shape[:2], dtype=np.uint8)

            obstacle_mask = (all_objects_mask > 0) & ~(target_object_mask > 0)

            # --- 3. 生成点云并进行预处理 ---
            self.get_logger().info("步骤 3/5: 从深度图生成点云并进行过滤...")
            depth_cv = self.bridge.imgmsg_to_cv2(self.latest_depth_image, "passthrough")
            cam_info = self.latest_camera_info
            fx, fy, cx, cy = cam_info.k[0], cam_info.k[4], cam_info.k[2], cam_info.k[5]
            graspnet_cam = GraspNetCameraInfo(depth_cv.shape[1], depth_cv.shape[0], fx, fy, cx, cy, 1000.0)
            full_cloud = create_point_cloud_from_depth_image(depth_cv, graspnet_cam, organized=True)
            
            if obstacle_mask.shape != depth_cv.shape:
                obstacle_mask_resized = cv2.resize(obstacle_mask.astype(np.uint8), (depth_cv.shape[1], depth_cv.shape[0]), interpolation=cv2.INTER_NEAREST)
            else:
                obstacle_mask_resized = obstacle_mask.astype(np.uint8)

            final_obstacle_mask = (obstacle_mask_resized > 0) & (depth_cv > 0)
            obstacle_points = full_cloud[final_obstacle_mask]
            
            if len(obstacle_points) == 0:
                self.get_logger().info("在分割区域内未找到有效的障碍物点云。")
                self.clear_markers()
                response.success = True
                response.message = "未找到障碍物。"
                return response
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(obstacle_points)
            self.get_logger().info(f"  - 初始障碍物点数: {len(pcd.points)}")

            # --- 新增步骤: 工作空间过滤 (ROI Filtering) ---
            # !!! 关键: 您需要根据您机器人的实际工作空间来调整这些值 !!!
            # 这些坐标是在 `camera_depth_frame` 坐标系下的。
            # X: 左右, Y: 上下, Z: 前后 (距离相机)
            min_bound = np.array([-0.5, -0.4, 0.1])  # x, y, z
            max_bound = np.array([0.5, 0.4, 1.0])   # x, y, z
            bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
            pcd = pcd.crop(bbox)
            self.get_logger().info(f"  - 工作空间过滤后点数: {len(pcd.points)}")

            # --- 新增步骤: 体素下采样 (Voxel Downsampling) ---
            voxel_size = 0.005  # 5mm的立方体
            pcd = pcd.voxel_down_sample(voxel_size)
            self.get_logger().info(f"  - 体素下采样后点数: {len(pcd.points)}")
            
            if len(pcd.points) < 10: # 如果过滤后点太少，就没必要聚类了
                 self.get_logger().info("过滤后点数过少，无障碍物需要处理。")
                 self.clear_markers()
                 response.success = True
                 response.message = "过滤后无障碍物。"
                 return response
                 
            # --- 4. 聚类 (现在是在过滤后的点云上进行) ---
            self.get_logger().info(f"步骤 4/5: 对 {len(pcd.points)} 个障碍物点进行聚类...")
            labels = np.array(pcd.cluster_dbscan(eps=0.04, min_points=20, print_progress=True))
            
            max_label = labels.max()
            noise_points = np.sum(np.array(labels) == -1)
            self.get_logger().info(f"聚类完成，发现 {max_label + 1} 个障碍物。另外有 {noise_points} 个点被认为是噪声。")
            
            # --- 5. 生成凸包 (这部分代码不变) ---
            self.get_logger().info("步骤 5/5: 生成凸包并发布可视化标记...")
            marker_array = MarkerArray()
            clear_marker = Marker()
            clear_marker.header.frame_id = self.camera_frame
            clear_marker.action = Marker.DELETEALL
            marker_array.markers.append(clear_marker)

            for i in range(max_label + 1):
                cluster_indices = np.where(labels == i)[0]
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
        """将Open3D的TriangleMesh转换为ROS的Marker"""
        marker = Marker()
        marker.header.frame_id = self.camera_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "obstacle_hulls"
        marker.id = marker_id
        marker.type = Marker.TRIANGLE_LIST # 使用三角面片列表来显示实体
        marker.action = Marker.ADD
        
        # 设置姿态
        marker.pose.orientation.w = 1.0
        
        # 设置尺寸
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        
        # 设置颜色（半透明红色）
        marker.color = ColorRGBA(r=0.5, g=0.0, b=0.8, a=0.4)
        
        # 获取所有的顶点和所有的三角面片索引
        vertices = np.asarray(hull.vertices)
        triangles = np.asarray(hull.triangles)

        # 遍历每个三角面片
        for triangle_indices in triangles:
            # 对每个三角面片，我们将其三个顶点按顺序添加到 marker.points 列表
            # triangle_indices 是一个包含三个数字的数组，例如 [5, 12, 34]，
            # 分别是该三角形三个顶点在 vertices 数组中的索引。
            
            # 添加第一个顶点
            p1 = Point()
            p1.x, p1.y, p1.z = vertices[triangle_indices[0]]
            marker.points.append(p1)
            
            # 添加第二个顶点
            p2 = Point()
            p2.x, p2.y, p2.z = vertices[triangle_indices[1]]
            marker.points.append(p2)
            
            # 添加第三个顶点
            p3 = Point()
            p3.x, p3.y, p3.z = vertices[triangle_indices[2]]
            marker.points.append(p3)

        return marker

    def clear_markers(self):
        """发布一个空的MarkerArray来清除RViz中的所有旧标记"""
        marker_array = MarkerArray()
        clear_marker = Marker()
        clear_marker.header.frame_id = self.camera_frame
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)
        self.marker_pub.publish(marker_array)
        self.get_logger().info("已清除所有障碍物标记。")

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