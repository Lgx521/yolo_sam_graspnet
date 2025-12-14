#!/usr/bin/env python3

import numpy as np
from typing import List

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Quaternion, Vector3, Pose
from std_msgs.msg import ColorRGBA
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2

from scipy.spatial.transform import Rotation as R


class GraspVisualizer(Node):
    """ROS2 node for visualizing grasps and point clouds"""
    
    def __init__(self):
        super().__init__('grasp_visualizer')
        
        # Publishers
        self.grasp_marker_pub = self.create_publisher(
            MarkerArray,
            'grasp_markers',
            10
        )
        
        self.point_cloud_pub = self.create_publisher(
            PointCloud2,
            'grasp_point_cloud',
            10
        )
        
        # Marker ID counter
        self.marker_id = 0
        
        self.get_logger().info('Grasp visualizer initialized')
    
    def create_gripper_marker(self, pose: Pose, width: float, 
                             score: float, marker_id: int,
                             frame_id: str = 'base_link') -> List[Marker]:
        """
        Create gripper visualization markers
        
        Args:
            pose: Gripper pose
            width: Gripper opening width
            score: Grasp quality score
            marker_id: Base marker ID
            frame_id: Reference frame
            
        Returns:
            List of markers representing the gripper
        """
        markers = []
        
        # Color based on score (red to green)
        color = ColorRGBA()
        color.r = 1.0 - score
        color.g = score
        color.b = 0.0
        color.a = 0.8
        
        # Gripper base (palm)
        palm_marker = Marker()
        palm_marker.header.frame_id = frame_id
        palm_marker.header.stamp = self.get_clock().now().to_msg()
        palm_marker.ns = "gripper_palm"
        palm_marker.id = marker_id
        palm_marker.type = Marker.CUBE
        palm_marker.action = Marker.ADD
        
        palm_marker.pose = pose
        palm_marker.scale = Vector3(x=0.02, y=0.08, z=0.04)
        palm_marker.color = color
        
        markers.append(palm_marker)
        
        # Gripper fingers
        finger_length = 0.08
        finger_thickness = 0.01
        
        # Convert pose to transformation matrix
        q = pose.orientation
        rotation = R.from_quat([q.x, q.y, q.z, q.w])
        rotation_matrix = rotation.as_matrix()
        
        position = np.array([pose.position.x, pose.position.y, pose.position.z])
        
        # Left finger
        left_finger = Marker()
        left_finger.header = palm_marker.header
        left_finger.ns = "gripper_left_finger"
        left_finger.id = marker_id + 1
        left_finger.type = Marker.CUBE
        left_finger.action = Marker.ADD
        
        # Position relative to palm
        left_offset = rotation_matrix @ np.array([0, width/2, finger_length/2])
        left_pos = position + left_offset
        
        left_finger.pose.position = Point(x=left_pos[0], y=left_pos[1], z=left_pos[2])
        left_finger.pose.orientation = pose.orientation
        left_finger.scale = Vector3(x=finger_thickness, y=finger_thickness, z=finger_length)
        left_finger.color = color
        
        markers.append(left_finger)
        
        # Right finger
        right_finger = Marker()
        right_finger.header = palm_marker.header
        right_finger.ns = "gripper_right_finger"
        right_finger.id = marker_id + 2
        right_finger.type = Marker.CUBE
        right_finger.action = Marker.ADD
        
        # Position relative to palm
        right_offset = rotation_matrix @ np.array([0, -width/2, finger_length/2])
        right_pos = position + right_offset
        
        right_finger.pose.position = Point(x=right_pos[0], y=right_pos[1], z=right_pos[2])
        right_finger.pose.orientation = pose.orientation
        right_finger.scale = Vector3(x=finger_thickness, y=finger_thickness, z=finger_length)
        right_finger.color = color
        
        markers.append(right_finger)
        
        # Approach vector arrow
        approach_marker = Marker()
        approach_marker.header = palm_marker.header
        approach_marker.ns = "approach_vector"
        approach_marker.id = marker_id + 3
        approach_marker.type = Marker.ARROW
        approach_marker.action = Marker.ADD
        
        # Arrow points from pre-grasp to grasp position
        approach_length = 0.1
        approach_vector = -rotation_matrix[:, 2]  # -Z axis
        approach_start = position + approach_vector * approach_length
        
        approach_marker.points = [
            Point(x=approach_start[0], y=approach_start[1], z=approach_start[2]),
            Point(x=position[0], y=position[1], z=position[2])
        ]
        
        approach_marker.scale = Vector3(x=0.005, y=0.01, z=0.01)
        approach_marker.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.8)
        
        markers.append(approach_marker)
        
        # Score text
        score_marker = Marker()
        score_marker.header = palm_marker.header
        score_marker.ns = "grasp_score"
        score_marker.id = marker_id + 4
        score_marker.type = Marker.TEXT_VIEW_FACING
        score_marker.action = Marker.ADD
        
        text_offset = rotation_matrix @ np.array([0, 0, -0.05])
        text_pos = position + text_offset
        
        score_marker.pose.position = Point(x=text_pos[0], y=text_pos[1], z=text_pos[2])
        score_marker.pose.orientation = pose.orientation
        score_marker.scale.z = 0.02
        score_marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        score_marker.text = f"{score:.2f}"
        
        markers.append(score_marker)
        
        return markers
    
    def publish_grasp_markers(self, grasp_poses: List[Pose], 
                             grasp_widths: List[float],
                             grasp_scores: List[float],
                             frame_id: str = 'base_link'):
        """
        Publish grasp visualization markers
        
        Args:
            grasp_poses: List of grasp poses
            grasp_widths: List of gripper widths
            grasp_scores: List of grasp quality scores
            frame_id: Reference frame
        """
        marker_array = MarkerArray()
        
        # Clear previous markers
        clear_marker = Marker()
        clear_marker.header.frame_id = frame_id
        clear_marker.header.stamp = self.get_clock().now().to_msg()
        clear_marker.ns = ""
        clear_marker.id = 0
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)
        
        # Add grasp markers
        marker_id = 0
        for pose, width, score in zip(grasp_poses, grasp_widths, grasp_scores):
            gripper_markers = self.create_gripper_marker(
                pose, width, score, marker_id, frame_id
            )
            marker_array.markers.extend(gripper_markers)
            marker_id += 5  # Each gripper uses 5 markers
        
        self.grasp_marker_pub.publish(marker_array)
        self.get_logger().info(f'Published {len(grasp_poses)} grasp markers')
    
    def publish_point_cloud(self, points: np.ndarray, 
                           colors: np.ndarray = None,
                           frame_id: str = 'camera_depth_frame'):
        """
        Publish point cloud
        
        Args:
            points: Nx3 array of points
            colors: Optional Nx3 array of RGB colors (0-1 range)
            frame_id: Reference frame
        """
        header = self.get_clock().now().to_msg()
        
        # Create fields
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        
        if colors is not None:
            fields.append(
                PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)
            )
            
            # Convert colors to packed RGB
            rgb_packed = []
            for color in colors:
                r = int(color[0] * 255)
                g = int(color[1] * 255)
                b = int(color[2] * 255)
                rgb = (r << 16) | (g << 8) | b
                rgb_packed.append(rgb)
            
            # Combine points and colors
            cloud_data = []
            for point, rgb in zip(points, rgb_packed):
                cloud_data.append([point[0], point[1], point[2], rgb])
        else:
            cloud_data = points
        
        # Create PointCloud2 message
        cloud_msg = point_cloud2.create_cloud(
            header,
            fields,
            cloud_data
        )
        cloud_msg.header.frame_id = frame_id
        
        self.point_cloud_pub.publish(cloud_msg)
        self.get_logger().info(f'Published point cloud with {len(points)} points')
    
    def create_workspace_marker(self, min_point: Point, max_point: Point,
                               frame_id: str = 'base_link') -> Marker:
        """Create workspace boundary visualization"""
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "workspace"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        
        # Define cube vertices
        vertices = [
            [min_point.x, min_point.y, min_point.z],
            [max_point.x, min_point.y, min_point.z],
            [max_point.x, max_point.y, min_point.z],
            [min_point.x, max_point.y, min_point.z],
            [min_point.x, min_point.y, max_point.z],
            [max_point.x, min_point.y, max_point.z],
            [max_point.x, max_point.y, max_point.z],
            [min_point.x, max_point.y, max_point.z]
        ]
        
        # Define edges
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
            (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
        ]
        
        # Create line list
        for edge in edges:
            p1 = vertices[edge[0]]
            p2 = vertices[edge[1]]
            marker.points.append(Point(x=p1[0], y=p1[1], z=p1[2]))
            marker.points.append(Point(x=p2[0], y=p2[1], z=p2[2]))
        
        marker.scale.x = 0.002  # Line width
        marker.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.5)
        
        return marker


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = GraspVisualizer()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()