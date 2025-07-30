#!/usr/bin/env python3

import os
import sys
import warnings
import contextlib
from io import StringIO

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ['YOLO_VERBOSE'] = 'False'

# Add graspnet-baseline to Python path
sys.path.insert(0, '/home/roar/graspnet/graspnet-baseline')
sys.path.insert(0, '/home/roar/graspnet/graspnet-baseline/models')
sys.path.insert(0, '/home/roar/graspnet/graspnet-baseline/utils')

import numpy as np
import cv2
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# Import the segmentation module
sys.path.append('/home/roar/graspnet/graspnet-baseline/kinova_graspnet_ros2/utils')



# Context manager to suppress stdout/stderr
@contextlib.contextmanager
def suppress_stdout_stderr():
    """Context manager to suppress stdout and stderr"""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


class YoloDetectionNode(Node):
    """实时YOLO检测节点，持续发布检测可视化结果"""
    
    def __init__(self):
        super().__init__('yolo_detection_node')
        
        # Declare parameters
        self.declare_parameter('color_image_topic', '/camera/color/image_raw')
        self.declare_parameter('target_object_class', '')  # 空字符串表示检测所有对象
        self.declare_parameter('detection_fps', 2.0)  # 检测频率 (Hz)
        self.declare_parameter('confidence_threshold', 0.25)
        self.declare_parameter('verbose', False)  # 新增参数控制是否输出详细信息
        
        # Get parameters
        self.color_image_topic = self.get_parameter('color_image_topic').value
        self.target_object_class = self.get_parameter('target_object_class').value
        self.detection_fps = self.get_parameter('detection_fps').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.verbose = self.get_parameter('verbose').value
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Initialize YOLO detector with suppressed output
        with suppress_stdout_stderr():
            from cv_segmentation import SmartSegmentation
            self.detector = SmartSegmentation()
        
        # Subscriber for RGB images
        self.image_sub = self.create_subscription(
            Image,
            self.color_image_topic,
            self.image_callback,
            10
        )
        
        # Publisher for YOLO detection visualization
        self.detection_pub = self.create_publisher(
            Image,
            'yolo_detection_visualization',
            10
        )
        
        # Store latest image
        self.latest_image = None
        self.latest_image_time = None
        self.image_received = False  # 标记是否已收到过图像
        
        # Create timer for periodic detection
        detection_period = 1.0 / self.detection_fps
        self.detection_timer = self.create_timer(
            detection_period,
            self.detection_callback
        )
        
        # 只在启动时输出一次信息
        self.get_logger().info(f'YOLO实时检测节点已启动')
        if self.verbose:
            self.get_logger().info(f'  订阅主题: {self.color_image_topic}')
            self.get_logger().info(f'  发布主题: /yolo_detection_visualization')
            self.get_logger().info(f'  检测频率: {self.detection_fps} Hz')
            self.get_logger().info(f'  目标对象: {"所有对象" if not self.target_object_class else self.target_object_class}')
            self.get_logger().info(f'  置信度阈值: {self.confidence_threshold}')
    
    def image_callback(self, msg: Image):
        """接收RGB图像回调"""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, msg.encoding)
            self.latest_image = cv_image
            self.latest_image_time = msg.header.stamp
            
            # 只在第一次收到图像时输出日志
            if not self.image_received:
                self.image_received = True
                if self.verbose:
                    self.get_logger().info('相机图像已连接，开始检测')
                    
        except Exception as e:
            self.get_logger().error(f'图像转换失败: {e}')
    
    def detection_callback(self):
        """定时检测回调"""
        if self.latest_image is None:
            # 改为debug级别，避免持续输出警告
            self.get_logger().debug('等待相机图像...')
            return
        
        try:
            # 使用当前图像进行YOLO检测，完全抑制输出
            target_class = self.target_object_class if self.target_object_class else None
            
            # 完全抑制YOLO模型的输出
            with suppress_stdout_stderr():
                detections, vis_img = self.detector.detect_objects(
                    self.latest_image,
                    target_class=target_class,
                    conf_threshold=self.confidence_threshold
                )
            
            # 发布可视化图像
            if vis_img is not None:
                try:
                    detection_msg = self.bridge.cv2_to_imgmsg(vis_img, vis_img.encoding)
                    detection_msg.header.stamp = self.latest_image_time if self.latest_image_time else self.get_clock().now().to_msg()
                    detection_msg.header.frame_id = "camera_color_frame"
                    
                    self.detection_pub.publish(detection_msg)
                    
                    # 完全静默发布，不打印任何检测信息
                    # 如果需要调试信息，可以设置 verbose=True 参数
                    if self.verbose:
                        self.get_logger().debug(f'检测到 {len(detections)} 个对象')
                        
                except Exception as e:
                    self.get_logger().error(f'发布检测结果失败: {e}')
            
        except Exception as e:
            # 只在verbose模式下输出详细错误，否则仅记录到debug
            if self.verbose:
                self.get_logger().error(f'YOLO检测失败: {e}')
                import traceback
                self.get_logger().error(f'错误详情: {traceback.format_exc()}')
            else:
                self.get_logger().debug(f'YOLO检测失败: {e}')


def main(args=None):
    # Suppress all warnings and YOLO output
    warnings.filterwarnings("ignore")
    os.environ['ULTRALYTICS_SILENT'] = 'true'
    
    rclpy.init(args=args)
    
    try:
        node = YoloDetectionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()