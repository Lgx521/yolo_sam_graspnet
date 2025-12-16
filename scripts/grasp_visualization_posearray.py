#!/usr/bin/env python3
"""
抓取可视化节点 - 使用 PoseArray 在 RViz 中显示抓取姿态
将相机坐标系的抓取姿态转换到 base_link 坐标系并发布
this is a new file created for visualization
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time

from geometry_msgs.msg import PoseArray, PoseStamped, Pose
from std_msgs.msg import Header
from std_srvs.srv import Trigger

import tf2_ros
from tf2_ros import TransformException
from scipy.spatial.transform import Rotation as R

# Import custom service definitions
from kinova_graspnet_ros2.srv import DetectGrasps


class GraspVisualizationPoseArray(Node):
    """
    抓取可视化节点 - 发布 PoseArray 到 RViz
    自动订阅检测服务，转换坐标系，发布到 base_link
    """
    
    def __init__(self):
        super().__init__('grasp_visualization_posearray')
        
        # Declare parameters
        self.declare_parameter('base_frame', 'base_link')
        # self.declare_parameter('camera_frame', 'camera_depth_optical_frame')  # GraspNet uses optical frame
        self.declare_parameter('camera_frame', 'camera_color_frame')  # GraspNet uses optical frame
        self.declare_parameter('max_display_grasps', 10)
        
        # Get parameters
        self.base_frame = self.get_parameter('base_frame').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.max_display_grasps = self.get_parameter('max_display_grasps').value
        
        # Initialize TF2
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Publisher for PoseArray visualization
        self.pose_array_pub = self.create_publisher(
            PoseArray,
            'grasp_poses_visualization',
            10
        )
        
        # Service client for grasp detection
        self.grasp_client = self.create_client(
            DetectGrasps,
            'detect_grasps'
        )
        
        # Service for triggering visualization
        self.visualize_srv = self.create_service(
            Trigger,
            'trigger_grasp_visualization',
            self.visualize_callback
        )
        
        # Wait for grasp detection service
        self.get_logger().info('等待抓取检测服务...')
        if self.grasp_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().info('✅ 抓取检测服务已连接')
        else:
            self.get_logger().warn('⚠️ 抓取检测服务未找到，将在需要时等待')
        
        self.get_logger().info(f'抓取可视化节点已初始化 (PoseArray mode)')
        self.get_logger().info(f'  - 目标坐标系: {self.base_frame}')
        self.get_logger().info(f'  - 源坐标系: {self.camera_frame}')
        self.get_logger().info(f'  - 发布话题: /grasp_poses_visualization')
        self.get_logger().info(f'  - 触发服务: /trigger_grasp_visualization')
    
    def transform_pose_to_base_link(self, pose_stamped: PoseStamped) -> PoseStamped:
        """
        将姿态从相机坐标系转换到 base_link 坐标系
        使用项目标准的 TF2 转换逻辑
        """
        try:
            # 获取从源坐标系到目标坐标系的变换
            transform = self.tf_buffer.lookup_transform(
                self.base_frame,  # target frame
                pose_stamped.header.frame_id,  # source frame
                Time(),  # 最新的变换
                timeout=Duration(seconds=2.0)
            )
            
            # 提取相机到基座的变换矩阵
            t = transform.transform.translation
            r = transform.transform.rotation
            
            # 构建 T_base_camera (4x4)
            T_base_camera = np.eye(4)
            T_base_camera[:3, 3] = [t.x, t.y, t.z]
            T_base_camera[:3, :3] = R.from_quat([r.x, r.y, r.z, r.w]).as_matrix()
            
            # 提取抓取姿态
            pos = np.array([
                pose_stamped.pose.position.x,
                pose_stamped.pose.position.y,
                pose_stamped.pose.position.z
            ])
            quat = np.array([
                pose_stamped.pose.orientation.x,
                pose_stamped.pose.orientation.y,
                pose_stamped.pose.orientation.z,
                pose_stamped.pose.orientation.w
            ])
            
            # 构建 T_camera_grasp (4x4)
            T_camera_grasp = np.eye(4)
            T_camera_grasp[:3, 3] = pos
            T_camera_grasp[:3, :3] = R.from_quat(quat).as_matrix()
            
            # 计算 T_base_grasp = T_base_camera @ T_camera_grasp
            T_base_grasp = T_base_camera @ T_camera_grasp
            
            # 转换回 PoseStamped
            result = PoseStamped()
            result.header.frame_id = self.base_frame
            result.header.stamp = self.get_clock().now().to_msg()
            
            # 位置
            result.pose.position.x = float(T_base_grasp[0, 3])
            result.pose.position.y = float(T_base_grasp[1, 3])
            result.pose.position.z = float(T_base_grasp[2, 3])
            
            # 方向
            rot = R.from_matrix(T_base_grasp[:3, :3])
            quat_result = rot.as_quat()  # [x, y, z, w]
            result.pose.orientation.x = float(quat_result[0])
            result.pose.orientation.y = float(quat_result[1])
            result.pose.orientation.z = float(quat_result[2])
            result.pose.orientation.w = float(quat_result[3])
            
            return result
            
        except TransformException as e:
            self.get_logger().error(f'❌ 坐标转换失败: {e}')
            return None
    
    def publish_grasp_poses(self, grasp_poses: list) -> bool:
        """
        发布抓取姿态为 PoseArray 消息
        
        Args:
            grasp_poses: 相机坐标系下的 PoseStamped 列表
            
        Returns:
            是否成功发布
        """
        if not grasp_poses:
            self.get_logger().warn('⚠️ 没有抓取姿态可以发布')
            return False
        
        # 创建 PoseArray 消息
        pose_array = PoseArray()
        pose_array.header.frame_id = self.base_frame
        pose_array.header.stamp = self.get_clock().now().to_msg()
        
        # 转换所有抓取姿态到 base_link
        transformed_count = 0
        display_count = min(len(grasp_poses), self.max_display_grasps)
        
        self.get_logger().info(f'🔄 开始转换坐标系: {len(grasp_poses)} 个抓取 -> base_link')
        
        for i, pose_stamped in enumerate(grasp_poses[:display_count]):
            # 转换坐标系
            transformed_pose = self.transform_pose_to_base_link(pose_stamped)
            
            if transformed_pose is not None:
                pose_array.poses.append(transformed_pose.pose)
                transformed_count += 1
                
                # 打印前3个抓取的详细信息
                if i < 3:
                    pos = transformed_pose.pose.position
                    self.get_logger().info(
                        f'  抓取 #{i+1}: 位置=({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f})'
                    )
        
        # 发布 PoseArray
        if transformed_count > 0:
            self.pose_array_pub.publish(pose_array)
            self.get_logger().info(
                f'✅ 已发布 {transformed_count}/{display_count} 个抓取姿态到 '
                f'/grasp_poses_visualization'
            )
            return True
        else:
            self.get_logger().error('❌ 所有坐标转换都失败了')
            return False
    
    def visualize_callback(self, request, response):
        """
        服务回调 - 触发抓取检测和可视化
        """
        try:
            self.get_logger().info('🔍 收到可视化请求，调用抓取检测服务...')
            
            # 调用抓取检测服务
            grasp_request = DetectGrasps.Request()
            grasp_request.target_object_class = 'bottle'  # 默认检测瓶子
            grasp_request.max_grasps = self.max_display_grasps
            
            # 等待服务可用
            if not self.grasp_client.wait_for_service(timeout_sec=2.0):
                response.success = False
                response.message = '抓取检测服务不可用'
                return response
            
            # 同步调用服务
            future = self.grasp_client.call_async(grasp_request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=30.0)
            
            if future.result() is None:
                response.success = False
                response.message = '抓取检测服务调用失败'
                return response
            
            grasp_response = future.result()
            
            if not grasp_response.success or len(grasp_response.grasp_poses) == 0:
                response.success = False
                response.message = f'抓取检测失败: {grasp_response.message}'
                return response
            
            # 发布可视化
            success = self.publish_grasp_poses(grasp_response.grasp_poses)
            
            if success:
                response.success = True
                response.message = f'成功可视化 {len(grasp_response.grasp_poses)} 个抓取姿态'
            else:
                response.success = False
                response.message = '坐标转换失败'
            
        except Exception as e:
            self.get_logger().error(f'❌ 可视化过程出错: {str(e)}')
            response.success = False
            response.message = f'错误: {str(e)}'
        
        return response
    
    def visualize_from_detection_response(self, detection_response) -> bool:
        """
        直接从检测响应发布可视化
        这个方法可以被外部调用，不需要重新调用检测服务
        
        Args:
            detection_response: DetectGrasps.Response 对象
            
        Returns:
            是否成功发布
        """
        if not detection_response.success or len(detection_response.grasp_poses) == 0:
            self.get_logger().warn('⚠️ 检测响应无效或没有抓取')
            return False
        
        self.get_logger().info(
            f'📊 收到 {len(detection_response.grasp_poses)} 个抓取姿态，准备可视化'
        )
        
        return self.publish_grasp_poses(detection_response.grasp_poses)


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = GraspVisualizationPoseArray()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
