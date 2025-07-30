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
    Publishes the grasp_center frame based on robotiq finger tip frames.
    
    The grasp_center frame:
    - Has the same orientation as the finger tip frames
    - Is positioned at the midpoint between left and right finger tips
    - Is offset 3cm forward along the Z-axis (gripper approach direction)
    """
    
    def __init__(self):
        super().__init__('grasp_center_publisher')
        
        # Declare parameters
        self.declare_parameter('left_finger_frame', 'robotiq_85_left_finger_tip_link')
        self.declare_parameter('right_finger_frame', 'robotiq_85_right_finger_tip_link')
        self.declare_parameter('grasp_center_frame', 'grasp_center')
        self.declare_parameter('z_offset', 0.03)  # 3cm offset along Z axis
        self.declare_parameter('publish_rate', 50.0)  # Hz
        
        # Get parameters
        self.left_finger_frame = self.get_parameter('left_finger_frame').value
        self.right_finger_frame = self.get_parameter('right_finger_frame').value
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
        self.get_logger().info(f'  Left finger: {self.left_finger_frame}')
        self.get_logger().info(f'  Right finger: {self.right_finger_frame}')
        self.get_logger().info(f'  Grasp center: {self.grasp_center_frame}')
        self.get_logger().info(f'  Z offset: {self.z_offset}m')
    
    def publish_grasp_center(self):
        """Compute and publish grasp_center frame"""
        try:
            # Get transforms for both finger tips relative to base_link
            # This ensures we have a common reference frame
            left_transform = self.tf_buffer.lookup_transform(
                'base_link',
                self.left_finger_frame,
                rclpy.time.Time()
            )
            
            right_transform = self.tf_buffer.lookup_transform(
                'base_link',
                self.right_finger_frame,
                rclpy.time.Time()
            )
            
            # Extract positions
            left_pos = np.array([
                left_transform.transform.translation.x,
                left_transform.transform.translation.y,
                left_transform.transform.translation.z
            ])
            
            right_pos = np.array([
                right_transform.transform.translation.x,
                right_transform.transform.translation.y,
                right_transform.transform.translation.z
            ])
            
            # Calculate midpoint
            midpoint = (left_pos + right_pos) / 2.0
            
            # Use the orientation from left finger (they should be the same)
            # Extract rotation as quaternion
            q = left_transform.transform.rotation
            rotation = R.from_quat([q.x, q.y, q.z, q.w])
            rotation_matrix = rotation.as_matrix()
            
            # Apply Z-axis offset (forward along gripper Z-axis)
            # In the finger frame, Z points forward (approach direction)
            z_axis = rotation_matrix[:, 2]
            grasp_center_pos = midpoint + z_axis * self.z_offset
            
            # Create and publish transform from base_link to grasp_center
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = 'base_link'
            t.child_frame_id = self.grasp_center_frame
            
            # Set translation
            t.transform.translation.x = grasp_center_pos[0]
            t.transform.translation.y = grasp_center_pos[1]
            t.transform.translation.z = grasp_center_pos[2]
            
            # Set rotation (same as finger tips)
            t.transform.rotation = left_transform.transform.rotation
            
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