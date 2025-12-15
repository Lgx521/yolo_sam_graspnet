#!/usr/bin/env python3
"""
抓取检测客户端 - 使用RGBD数据进行抓取检测
"""

import rclpy
from rclpy.node import Node
import sys

from kinova_graspnet_ros2.srv import DetectGrasps
from geometry_msgs.msg import Point, PoseArray
from std_srvs.srv import Trigger


class DetectGraspsClient(Node):
    """简化的抓取检测客户端，输出清晰的结果"""
    
    def __init__(self):
        super().__init__('detect_grasps_client')
        
        # 创建服务客户端
        self.client = self.create_client(DetectGrasps, 'detect_grasps')
        
        # 创建可视化发布器 (直接发布 PoseArray)
        self.pose_array_pub = self.create_publisher(
            PoseArray,
            'grasp_poses_visualization',
            10
        )
        
        # 等待服务可用
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('正在等待 detect_grasps 服务...')
        
        self.get_logger().info('✅ 连接到抓取检测服务')
        self.get_logger().info('✅ PoseArray 可视化已启用 (话题: /grasp_poses_visualization)')
    
    def call_detect_grasps(self, target_object_class='bottle', max_grasps=10):
        """调用抓取检测服务"""
        request = DetectGrasps.Request()
        request.target_object_class = target_object_class
        request.max_grasps = max_grasps
        
        # Set workspace constraints to focus on nearby objects (table-top area)
        # Based on point cloud analysis: focus on 20-60cm range
        request.workspace_min = Point()
        request.workspace_min.x = -0.4  # Left-right workspace
        request.workspace_min.y = -0.2  # Up-down workspace  
        request.workspace_min.z = 0.2   # Near distance (20cm from camera)
        
        request.workspace_max = Point()
        request.workspace_max.x = 0.4   # Right limit
        request.workspace_max.y = 0.4   # Down limit
        request.workspace_max.z = 0.6   # Far distance (60cm from camera)
        
        self.get_logger().info(f'🔍 开始检测抓取 (目标对象: {target_object_class}, 模式: RGBD模式)')
        
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        if future.result() is not None:
            response = future.result()
            self.display_results(response)
            
            # 自动发布可视化
            if response.success and len(response.grasp_poses) > 0:
                self.publish_visualization(response)
            
            return response
        else:
            self.get_logger().error('❌ 服务调用失败')
            return None
    
    def display_results(self, response):
        """显示清晰的结果"""
        if not response.success:
            print(f"\n❌ 检测失败: {response.message}")
            return
        
        if len(response.grasp_poses) == 0:
            print("\n❌ 未检测到有效抓取")
            return
            
        print("\n" + "="*100)
        print("🎯 抓取检测结果 - 前10名最佳抓取命令")
        print("="*100)
        
        top_n = min(10, len(response.grasp_poses))
        
        # 先显示摘要信息
        print(f"\n📊 检测摘要:")
        print(f"   - 总检测抓取数: {response.total_grasps_detected}")
        print(f"   - 过滤后抓取数: {response.grasps_after_filtering}")
        print(f"   - 推理时间: {response.inference_time:.3f}秒")
        print(f"   - 显示前{top_n}个最佳抓取\n")
        
        print("-"*100)
        print("可执行命令列表:")
        print("-"*100)
        
        for i in range(top_n):
            pose = response.grasp_poses[i]
            score = response.grasp_scores[i]
            width = response.grasp_widths[i]
            
            # 使用不同的表情符号来区分排名
            if i == 0:
                rank = "#1"
            elif i == 1:
                rank = "#2"
            elif i == 2:
                rank = "#3"
            else:
                rank = f"  #{i+1}"
            
            # 显示简要信息和命令
            print(f"\n{rank} [得分:{score:.3f}] [宽度:{width:.3f}m] [位置:({pose.pose.position.x:.3f},{pose.pose.position.y:.3f},{pose.pose.position.z:.3f})]")
            
            # 生成简化的单行命令格式 - 使用字符串格式化避免f-string的大括号问题
            pos_x = pose.pose.position.x
            pos_y = pose.pose.position.y
            pos_z = pose.pose.position.z
            ori_x = pose.pose.orientation.x
            ori_y = pose.pose.orientation.y
            ori_z = pose.pose.orientation.z
            ori_w = pose.pose.orientation.w
            frame_id = pose.header.frame_id
            
            command = f'ros2 service call /execute_grasp kinova_graspnet_ros2/srv/ExecuteGrasp "{{'
            command += f'grasp_pose: {{header: {{frame_id: \\"{frame_id}\\"}}, '
            command += f'pose: {{position: {{x: {pos_x}, y: {pos_y}, z: {pos_z}}}, '
            command += f'orientation: {{x: {ori_x}, y: {ori_y}, z: {ori_z}, w: {ori_w}}}}}}}, '
            command += f'grasp_width: {width}, approach_distance: -0.2, max_velocity_scaling: 0.15, max_acceleration_scaling: 0.15}}"'
            
            print(f"    {command}")
        
        print("\n" + "="*80)
        print(f"✅ 检测完成: 共{len(response.grasp_poses)}个抓取，耗时{response.inference_time:.3f}s")
        print("="*80)
    
    def publish_visualization(self, response):
        """
        发布 PoseArray 用于 RViz 可视化
        直接在相机坐标系下发布，让 RViz 通过 TF 自动转换
        """
        try:
            # 创建 PoseArray 消息
            pose_array = PoseArray()
            # 使用相机坐标系，让 RViz 通过 TF 自动转换到 base_link
            pose_array.header.frame_id = response.grasp_poses[0].header.frame_id
            pose_array.header.stamp = self.get_clock().now().to_msg()
            
            # 添加所有抓取姿态（限制为前10个）
            max_display = min(10, len(response.grasp_poses))
            for i in range(max_display):
                pose_array.poses.append(response.grasp_poses[i].pose)
            
            # 发布
            self.pose_array_pub.publish(pose_array)
            
            print(f"\n🎨 可视化信息:")
            print(f"   ✅ 已发布 {max_display} 个抓取姿态到 /grasp_poses_visualization")
            print(f"   📍 坐标系: {pose_array.header.frame_id}")
            print(f"   💡 在 RViz 中添加 PoseArray 话题即可查看")
            print(f"      - 话题: /grasp_poses_visualization")
            print(f"      - Fixed Frame: base_link (推荐) 或 {pose_array.header.frame_id}")
            
        except Exception as e:
            self.get_logger().error(f'❌ 发布可视化失败: {e}')


def main(args=None):
    rclpy.init(args=args)
    
    # 解析命令行参数
    target_object = 'bottle'
    max_grasps = 10  # 默认返回10个抓取
    
    # 显示使用说明
    if '--help' in sys.argv or '-h' in sys.argv:
        print("""使用方法:
    python3 detect_grasps_client.py [目标对象] [最大抓取数]
    
    参数:
        目标对象: 要抓取的对象类型 (默认: bottle)
        最大抓取数: 返回的最大抓取数量 (默认: 10)
    
    示例:
        python3 detect_grasps_client.py bottle          # 检测瓶子，返回10个抓取
        python3 detect_grasps_client.py cup 10          # 检测杯子，返回10个抓取
        python3 detect_grasps_client.py bottle 5        # 检测瓶子，返回5个抓取
        """)
        return
    
    if len(sys.argv) > 1 and not sys.argv[1].startswith('--'):
        target_object = sys.argv[1]
    if len(sys.argv) > 2 and not sys.argv[2].startswith('--'):
        try:
            max_grasps = int(sys.argv[2])
        except ValueError:
            print("⚠️ 最大抓取数量必须是整数，使用默认值 10")
    
    
    try:
        client = DetectGraspsClient()
        client.call_detect_grasps(target_object, max_grasps)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()