#!/usr/bin/env python3

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
import tf2_ros
from tf2_ros import TransformBroadcaster
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from scipy.spatial.transform import Rotation as R


class GraspCenterPublisher(Node):
    """
    Publishes the grasp_center frame based on tool_frame.
    
    For Kinova Gen3 Lite with integrated gripper:
    - tool_frame is already the gripper center point published by the robot
    - This node re-publishes it as grasp_center with optional offset
    """
    
    def __init__(self):
        super().__init__('grasp_center_publisher')
        
        # Declare parameters
        self.declare_parameter('source_frame', 'tool_frame')  # The frame to use as reference
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('grasp_center_frame', 'grasp_center')
        self.declare_parameter('z_offset', 0.0)  # Optional offset along Z axis (approach direction)
        self.declare_parameter('publish_rate', 50.0)  # Hz
        
        # Get parameters
        self.source_frame = self.get_parameter('source_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.grasp_center_frame = self.get_parameter('grasp_center_frame').value
        self.z_offset = self.get_parameter('z_offset').value
        self.publish_rate = self.get_parameter('publish_rate').value
        
        # Initialize TF2
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Create timer for publishing
        self.timer = self.create_timer(1.0 / self.publish_rate, self.publish_grasp_center)
        
        self.get_logger().info(f'Grasp center publisher initialized')
        self.get_logger().info(f'  Source frame: {self.source_frame}')
        self.get_logger().info(f'  Base frame: {self.base_frame}')
        self.get_logger().info(f'  Grasp center: {self.grasp_center_frame}')
        self.get_logger().info(f'  Z offset: {self.z_offset}m')
    
    def publish_grasp_center(self):
        """Compute and publish grasp_center frame based on tool_frame"""
        try:
            # Get transform from base_link to tool_frame
            tool_transform = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.source_frame,
                rclpy.time.Time()
            )
            
            # Extract position
            pos = np.array([
                tool_transform.transform.translation.x,
                tool_transform.transform.translation.y,
                tool_transform.transform.translation.z
            ])
            
            # Apply Z-axis offset if configured
            if abs(self.z_offset) > 0.001:
                q = tool_transform.transform.rotation
                rotation = R.from_quat([q.x, q.y, q.z, q.w])
                rotation_matrix = rotation.as_matrix()
                # Z-axis is the approach direction
                z_axis = rotation_matrix[:, 2]
                pos = pos + z_axis * self.z_offset
            
            # Create and publish transform from base_link to grasp_center
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = self.base_frame
            t.child_frame_id = self.grasp_center_frame
            
            # Set translation
            t.transform.translation.x = pos[0]
            t.transform.translation.y = pos[1]
            t.transform.translation.z = pos[2]
            
            # Set rotation (same as tool_frame)
            t.transform.rotation = tool_transform.transform.rotation
            
            # Publish transform
            self.tf_broadcaster.sendTransform(t)
            
        except tf2_ros.TransformException as e:
            # Don't spam logs - this is expected at startup
            pass
        except Exception as e:
            self.get_logger().error(f'Error computing grasp center: {e}', throttle_duration_sec=1.0)


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = GraspCenterPublisher()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()