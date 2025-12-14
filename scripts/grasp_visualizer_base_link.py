#!/usr/bin/env python3
"""
将抓取姿态转换到base_link坐标系并在RViz中可视化
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time

from geometry_msgs.msg import PoseStamped, Point, Quaternion, Pose
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import ColorRGBA, Header
from geometry_msgs.msg import Vector3

import tf2_ros
from tf2_ros import TransformException
from scipy.spatial.transform import Rotation as R

# Import custom service definitions
from kinova_graspnet_ros2.srv import DetectGrasps


class GraspVisualizerBaseLink(Node):
    """将抓取姿态可视化在base_link坐标系中"""
    
    def __init__(self):
        super().__init__('grasp_visualizer_base_link')
        
        # Initialize TF2
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Publishers
        self.marker_pub = self.create_publisher(
            MarkerArray,
            'grasp_markers_base_link',
            10
        )
        
        # Service client for grasp detection
        self.grasp_client = self.create_client(
            DetectGrasps,
            'detect_grasps'
        )
        
        # Service for manual visualization
        from std_srvs.srv import Trigger
        self.visualize_srv = self.create_service(
            Trigger,
            'visualize_grasps_base_link',
            self.visualize_grasps_callback
        )
        
        # Wait for services
        self.get_logger().info('Waiting for grasp detection service...')
        self.grasp_client.wait_for_service(timeout_sec=10.0)
        
        self.get_logger().info('Grasp visualizer (base_link) initialized')
        
    def transform_pose_to_base(self, pose_stamped: PoseStamped) -> PoseStamped:
        """Transform pose to base_link frame"""
        try:
            # Get transform
            transform = self.tf_buffer.lookup_transform(
                'base_link',
                pose_stamped.header.frame_id,
                Time(),
                timeout=Duration(seconds=1.0)
            )
            
            # Manual transformation using matrices for better control
            # Convert pose to transformation matrix
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
            
            # Get camera to base transform matrix
            t = transform.transform.translation
            r = transform.transform.rotation
            
            T_base_camera = np.eye(4)
            T_base_camera[:3, 3] = [t.x, t.y, t.z]
            T_base_camera[:3, :3] = R.from_quat([r.x, r.y, r.z, r.w]).as_matrix()
            
            # Create grasp transform matrix
            T_camera_grasp = np.eye(4)
            T_camera_grasp[:3, 3] = pos
            T_camera_grasp[:3, :3] = R.from_quat(quat).as_matrix()
            
            # Compute base transform
            T_base_grasp = T_base_camera @ T_camera_grasp
            
            # Convert back to PoseStamped
            result = PoseStamped()
            result.header.frame_id = 'base_link'
            result.header.stamp = self.get_clock().now().to_msg()
            
            # Position
            result.pose.position.x = float(T_base_grasp[0, 3])
            result.pose.position.y = float(T_base_grasp[1, 3])
            result.pose.position.z = float(T_base_grasp[2, 3])
            
            # Orientation
            rot = R.from_matrix(T_base_grasp[:3, :3])
            quat_result = rot.as_quat()  # [x, y, z, w]
            result.pose.orientation.x = float(quat_result[0])
            result.pose.orientation.y = float(quat_result[1])
            result.pose.orientation.z = float(quat_result[2])
            result.pose.orientation.w = float(quat_result[3])
            
            return result
            
        except TransformException as e:
            self.get_logger().error(f'Transform failed: {e}')
            return None
    
    def create_grasp_markers(self, grasp_poses_base: list, grasp_scores: list, grasp_widths: list) -> MarkerArray:
        """Create visualization markers for grasps in base_link frame"""
        marker_array = MarkerArray()
        
        # Clear previous markers
        clear_marker = Marker()
        clear_marker.header.frame_id = 'base_link'
        clear_marker.header.stamp = self.get_clock().now().to_msg()
        clear_marker.ns = ""
        clear_marker.id = 0
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)
        
        for i, (pose, score, width) in enumerate(zip(grasp_poses_base, grasp_scores, grasp_widths)):
            if i >= 10:  # Limit to first 10 grasps
                break
                
            markers = self.create_single_grasp_marker(pose.pose, width, score, i * 10, 'base_link')
            marker_array.markers.extend(markers)
        
        return marker_array
    
    def create_single_grasp_marker(self, pose: Pose, width: float, score: float, 
                                  marker_id: int, frame_id: str) -> list:
        """Create markers for a single grasp"""
        markers = []
        
        # Color based on score (red to green)
        color = ColorRGBA()
        color.r = max(0.0, 1.0 - score)
        color.g = min(1.0, score) 
        color.b = 0.0
        color.a = 0.8
        
        # Extract position and rotation
        position = np.array([pose.position.x, pose.position.y, pose.position.z])
        q = pose.orientation
        rotation = R.from_quat([q.x, q.y, q.z, q.w])
        rotation_matrix = rotation.as_matrix()
        
        # 1. Gripper center (cube)
        center_marker = Marker()
        center_marker.header.frame_id = frame_id
        center_marker.header.stamp = self.get_clock().now().to_msg()
        center_marker.ns = "grasp_center"
        center_marker.id = marker_id
        center_marker.type = Marker.CUBE
        center_marker.action = Marker.ADD
        center_marker.pose = pose
        center_marker.scale = Vector3(x=0.02, y=0.02, z=0.02)
        center_marker.color = color
        markers.append(center_marker)
        
        # 2. Left finger
        left_marker = Marker()
        left_marker.header = center_marker.header
        left_marker.ns = "left_finger"
        left_marker.id = marker_id + 1
        left_marker.type = Marker.CUBE
        left_marker.action = Marker.ADD
        
        # Finger offset (assuming X-axis is gripper opening direction)
        left_offset = rotation_matrix @ np.array([width/2, 0, 0.03])
        left_pos = position + left_offset
        left_marker.pose.position = Point(x=left_pos[0], y=left_pos[1], z=left_pos[2])
        left_marker.pose.orientation = pose.orientation
        left_marker.scale = Vector3(x=0.005, y=0.01, z=0.06)
        left_marker.color = color
        markers.append(left_marker)
        
        # 3. Right finger
        right_marker = Marker()
        right_marker.header = center_marker.header
        right_marker.ns = "right_finger"
        right_marker.id = marker_id + 2
        right_marker.type = Marker.CUBE
        right_marker.action = Marker.ADD
        
        right_offset = rotation_matrix @ np.array([-width/2, 0, 0.03])
        right_pos = position + right_offset
        right_marker.pose.position = Point(x=right_pos[0], y=right_pos[1], z=right_pos[2])
        right_marker.pose.orientation = pose.orientation
        right_marker.scale = Vector3(x=0.005, y=0.01, z=0.06)
        right_marker.color = color
        markers.append(right_marker)
        
        # 4. Approach vector (arrow)
        approach_marker = Marker()
        approach_marker.header = center_marker.header
        approach_marker.ns = "approach_vector"
        approach_marker.id = marker_id + 3
        approach_marker.type = Marker.ARROW
        approach_marker.action = Marker.ADD
        
        # Approach direction (negative Z-axis for safe approach)
        z_axis = rotation_matrix[:, 2]
        approach_vector = -z_axis  # Move away from object for approach
        approach_start = position + approach_vector * 0.08
        approach_end = position
        
        approach_marker.points = [
            Point(x=approach_start[0], y=approach_start[1], z=approach_start[2]),
            Point(x=approach_end[0], y=approach_end[1], z=approach_end[2])
        ]
        approach_marker.scale = Vector3(x=0.003, y=0.006, z=0.01)
        approach_marker.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.8)
        markers.append(approach_marker)
        
        # 5. Score text
        text_marker = Marker()
        text_marker.header = center_marker.header
        text_marker.ns = "score_text"
        text_marker.id = marker_id + 4
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        
        text_pos = position + np.array([0, 0, 0.04])
        text_marker.pose.position = Point(x=text_pos[0], y=text_pos[1], z=text_pos[2])
        text_marker.scale.z = 0.015
        text_marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        text_marker.text = f"G{marker_id//10}: {score:.2f}"
        markers.append(text_marker)
        
        # 6. Coordinate frame axes
        axis_length = 0.03
        
        # X-axis (red)
        x_marker = Marker()
        x_marker.header = center_marker.header
        x_marker.ns = "x_axis"
        x_marker.id = marker_id + 5
        x_marker.type = Marker.ARROW
        x_marker.action = Marker.ADD
        
        x_axis = rotation_matrix[:, 0]
        x_end = position + x_axis * axis_length
        x_marker.points = [
            Point(x=position[0], y=position[1], z=position[2]),
            Point(x=x_end[0], y=x_end[1], z=x_end[2])
        ]
        x_marker.scale = Vector3(x=0.002, y=0.004, z=0.006)
        x_marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.7)
        markers.append(x_marker)
        
        # Y-axis (green)
        y_marker = Marker()
        y_marker.header = center_marker.header
        y_marker.ns = "y_axis"
        y_marker.id = marker_id + 6
        y_marker.type = Marker.ARROW
        y_marker.action = Marker.ADD
        
        y_axis = rotation_matrix[:, 1]
        y_end = position + y_axis * axis_length
        y_marker.points = [
            Point(x=position[0], y=position[1], z=position[2]),
            Point(x=y_end[0], y=y_end[1], z=y_end[2])
        ]
        y_marker.scale = Vector3(x=0.002, y=0.004, z=0.006)
        y_marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.7)
        markers.append(y_marker)
        
        # Z-axis (blue) 
        z_marker = Marker()
        z_marker.header = center_marker.header
        z_marker.ns = "z_axis"
        z_marker.id = marker_id + 7
        z_marker.type = Marker.ARROW
        z_marker.action = Marker.ADD
        
        z_end = position + z_axis * axis_length
        z_marker.points = [
            Point(x=position[0], y=position[1], z=position[2]),
            Point(x=z_end[0], y=z_end[1], z=z_end[2])
        ]
        z_marker.scale = Vector3(x=0.002, y=0.004, z=0.006)
        z_marker.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.7)
        markers.append(z_marker)
        
        return markers
    
    def visualize_grasps_callback(self, request, response):
        """Service callback to visualize current grasps in base_link"""
        try:
            # Call grasp detection service
            grasp_request = DetectGrasps.Request()
            grasp_request.target_object_class = 'bottle'
            grasp_request.max_grasps = 5
            
            future = self.grasp_client.call_async(grasp_request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=30.0)
            
            if future.result() is None:
                response.success = False
                response.message = "Failed to call grasp detection service"
                return response
            
            grasp_response = future.result()
            
            if not grasp_response.success or len(grasp_response.grasp_poses) == 0:
                response.success = False
                response.message = f"Grasp detection failed: {grasp_response.message}"
                return response
            
            # Transform all grasp poses to base_link
            grasp_poses_base = []
            for pose in grasp_response.grasp_poses:
                transformed_pose = self.transform_pose_to_base(pose)
                if transformed_pose is not None:
                    grasp_poses_base.append(transformed_pose)
            
            if len(grasp_poses_base) == 0:
                response.success = False
                response.message = "Failed to transform any poses to base_link"
                return response
            
            # Create and publish markers
            markers = self.create_grasp_markers(
                grasp_poses_base,
                grasp_response.grasp_scores[:len(grasp_poses_base)],
                grasp_response.grasp_widths[:len(grasp_poses_base)]
            )
            
            self.marker_pub.publish(markers)
            
            # Log information
            self.get_logger().info(f'Published {len(grasp_poses_base)} grasp markers in base_link frame')
            for i, pose in enumerate(grasp_poses_base):
                pos = pose.pose.position
                self.get_logger().info(f'  Grasp {i}: ({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}) score: {grasp_response.grasp_scores[i]:.3f}')
            
            response.success = True
            response.message = f"Visualized {len(grasp_poses_base)} grasps in base_link frame"
            
        except Exception as e:
            self.get_logger().error(f'Visualization failed: {str(e)}')
            response.success = False
            response.message = f"Visualization error: {str(e)}"
        
        return response


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = GraspVisualizerBaseLink()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()