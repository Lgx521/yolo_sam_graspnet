#!/usr/bin/env python3

import numpy as np
from typing import Tuple, Optional
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TransformStamped
import tf2_ros
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster


class CoordinateTransformer(Node):
    """
    Coordinate transformation utilities for GraspNet + Kinova integration
    Handles transformations between camera, gripper, and base frames
    """
    
    def __init__(self):
        super().__init__('coordinate_transformer')
        
        # Declare parameters for hand-eye calibration
        self.declare_parameter('publish_static_transforms', True)
        self.declare_parameter('camera_to_ee_translation', [0.0, 0.0, 0.0])
        self.declare_parameter('camera_to_ee_rotation', [0.0, 0.0, 0.0, 1.0])  # quaternion
        
        # Get parameters
        self.publish_static = self.get_parameter('publish_static_transforms').value
        cam_to_ee_trans = self.get_parameter('camera_to_ee_translation').value
        cam_to_ee_rot = self.get_parameter('camera_to_ee_rotation').value
        
        # Initialize TF2
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.static_broadcaster = StaticTransformBroadcaster(self)
        
        # Publish static transform if configured
        if self.publish_static:
            self.publish_camera_to_ee_transform(cam_to_ee_trans, cam_to_ee_rot)
        
        self.get_logger().info('Coordinate transformer initialized')
    
    def publish_camera_to_ee_transform(self, translation: list, rotation_quat: list):
        """Publish static transform from camera to end effector"""
        static_transform = TransformStamped()
        static_transform.header.stamp = self.get_clock().now().to_msg()
        static_transform.header.frame_id = 'robotiq_85_base_link'
        static_transform.child_frame_id = 'camera_link'
        
        static_transform.transform.translation.x = translation[0]
        static_transform.transform.translation.y = translation[1]
        static_transform.transform.translation.z = translation[2]
        
        static_transform.transform.rotation.x = rotation_quat[0]
        static_transform.transform.rotation.y = rotation_quat[1]
        static_transform.transform.rotation.z = rotation_quat[2]
        static_transform.transform.rotation.w = rotation_quat[3]
        
        self.static_broadcaster.sendTransform(static_transform)
        self.get_logger().info('Published camera to end-effector transform')
    
    def grasp_to_gripper_transform(self, grasp_in_camera: np.ndarray) -> np.ndarray:
        """
        Transform grasp pose from camera frame to gripper frame
        
        Args:
            grasp_in_camera: 4x4 transformation matrix of grasp in camera frame
            
        Returns:
            4x4 transformation matrix of grasp in gripper frame
        """
        # GraspNet convention: grasp approach is along +Z, gripper closes along +X
        # Kinova convention: approach along -Z, gripper closes along +Y
        
        # Rotation to align GraspNet convention with Kinova
        # Rotate 180° around X to flip Z direction
        # Then rotate 90° around Z to align gripper closing direction
        R_align = R.from_euler('xz', [np.pi, np.pi/2]).as_matrix()
        
        # Apply alignment
        grasp_aligned = grasp_in_camera.copy()
        grasp_aligned[:3, :3] = grasp_in_camera[:3, :3] @ R_align
        
        return grasp_aligned
    
    def compute_pre_grasp_offset(self, grasp_pose: np.ndarray, offset_distance: float) -> np.ndarray:
        """
        Compute pre-grasp pose with offset along approach direction
        
        Args:
            grasp_pose: 4x4 transformation matrix
            offset_distance: Distance to offset along approach direction
            
        Returns:
            4x4 transformation matrix of pre-grasp pose
        """
        pre_grasp = grasp_pose.copy()
        
        # Approach vector is along -Z axis in gripper frame
        approach_vector = -grasp_pose[:3, 2]
        
        # Offset position
        pre_grasp[:3, 3] += approach_vector * offset_distance
        
        return pre_grasp
    
    def matrix_to_pose_stamped(self, matrix: np.ndarray, frame_id: str) -> PoseStamped:
        """Convert 4x4 transformation matrix to PoseStamped message"""
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = frame_id
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        
        # Extract translation
        pose_msg.pose.position.x = float(matrix[0, 3])
        pose_msg.pose.position.y = float(matrix[1, 3])
        pose_msg.pose.position.z = float(matrix[2, 3])
        
        # Extract rotation and convert to quaternion
        rotation = R.from_matrix(matrix[:3, :3])
        quat = rotation.as_quat()  # [x, y, z, w]
        
        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]
        
        return pose_msg
    
    def pose_stamped_to_matrix(self, pose: PoseStamped) -> np.ndarray:
        """Convert PoseStamped message to 4x4 transformation matrix"""
        matrix = np.eye(4)
        
        # Set translation
        matrix[0, 3] = pose.pose.position.x
        matrix[1, 3] = pose.pose.position.y
        matrix[2, 3] = pose.pose.position.z
        
        # Set rotation
        q = pose.pose.orientation
        rotation = R.from_quat([q.x, q.y, q.z, q.w])
        matrix[:3, :3] = rotation.as_matrix()
        
        return matrix
    
    def transform_grasp_to_base(self, grasp_in_camera: np.ndarray, 
                               camera_frame: str = 'camera_depth_frame',
                               base_frame: str = 'base_link') -> Optional[np.ndarray]:
        """
        Transform grasp from camera frame to robot base frame
        
        Args:
            grasp_in_camera: 4x4 transformation matrix in camera frame
            camera_frame: Camera frame ID
            base_frame: Base frame ID
            
        Returns:
            4x4 transformation matrix in base frame, or None if transform fails
        """
        try:
            # Get transform from camera to base
            transform = self.tf_buffer.lookup_transform(
                base_frame,
                camera_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
            
            # Convert transform to matrix
            T_base_to_camera = self.transform_to_matrix(transform)
            
            # Transform grasp to base frame
            grasp_in_base = T_base_to_camera @ grasp_in_camera
            
            return grasp_in_base
            
        except TransformException as e:
            self.get_logger().error(f'Transform lookup failed: {e}')
            return None
    
    def transform_to_matrix(self, transform: TransformStamped) -> np.ndarray:
        """Convert TransformStamped to 4x4 transformation matrix"""
        matrix = np.eye(4)
        
        # Translation
        t = transform.transform.translation
        matrix[0, 3] = t.x
        matrix[1, 3] = t.y
        matrix[2, 3] = t.z
        
        # Rotation
        q = transform.transform.rotation
        rotation = R.from_quat([q.x, q.y, q.z, q.w])
        matrix[:3, :3] = rotation.as_matrix()
        
        return matrix
    
    def validate_grasp_pose(self, grasp_pose: np.ndarray, 
                           workspace_limits: Optional[dict] = None) -> Tuple[bool, str]:
        """
        Validate if grasp pose is reachable and safe
        
        Args:
            grasp_pose: 4x4 transformation matrix in base frame
            workspace_limits: Optional dictionary with 'min' and 'max' keys
            
        Returns:
            (is_valid, reason): Tuple of validation result and reason
        """
        # Extract position
        position = grasp_pose[:3, 3]
        
        # Check workspace limits
        if workspace_limits:
            min_limits = workspace_limits.get('min', [-1, -1, 0])
            max_limits = workspace_limits.get('max', [1, 1, 1])
            
            for i, (pos, min_val, max_val) in enumerate(zip(position, min_limits, max_limits)):
                if pos < min_val or pos > max_val:
                    axis = ['x', 'y', 'z'][i]
                    return False, f"Position {axis}={pos:.3f} outside limits [{min_val}, {max_val}]"
        
        # Check orientation constraints
        # Ensure gripper is not pointing too far up (avoid singularities)
        z_axis = grasp_pose[:3, 2]
        if z_axis[2] > 0.9:  # Z-axis pointing up
            return False, "Grasp orientation too vertical (pointing up)"
        
        # Check for reasonable approach angle
        approach_angle = np.arccos(np.clip(-z_axis[2], -1, 1))
        if approach_angle > np.radians(150):  # More than 150 degrees from vertical
            return False, f"Approach angle {np.degrees(approach_angle):.1f}° too steep"
        
        return True, "Grasp pose is valid"


class HandEyeCalibrator(Node):
    """
    Utility for hand-eye calibration between camera and end-effector
    """
    
    def __init__(self):
        super().__init__('hand_eye_calibrator')
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Storage for calibration data
        self.ee_poses = []
        self.camera_poses = []
        
        self.get_logger().info('Hand-eye calibrator initialized')
    
    def collect_calibration_pair(self, marker_frame: str = 'marker',
                                base_frame: str = 'base_link',
                                ee_frame: str = 'end_effector_link',
                                camera_frame: str = 'camera_depth_frame') -> bool:
        """
        Collect a calibration data pair
        
        Args:
            marker_frame: Calibration marker frame
            base_frame: Robot base frame
            ee_frame: End-effector frame
            camera_frame: Camera frame
            
        Returns:
            True if data collection successful
        """
        try:
            # Get end-effector pose in base frame
            ee_transform = self.tf_buffer.lookup_transform(
                base_frame, ee_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
            
            # Get marker pose in camera frame
            camera_transform = self.tf_buffer.lookup_transform(
                camera_frame, marker_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
            
            # Convert to matrices and store
            ee_matrix = self.transform_to_matrix(ee_transform)
            camera_matrix = self.transform_to_matrix(camera_transform)
            
            self.ee_poses.append(ee_matrix)
            self.camera_poses.append(camera_matrix)
            
            self.get_logger().info(f'Collected calibration pair {len(self.ee_poses)}')
            return True
            
        except TransformException as e:
            self.get_logger().error(f'Failed to collect calibration data: {e}')
            return False
    
    def compute_calibration(self) -> Optional[np.ndarray]:
        """
        Compute hand-eye calibration using collected data
        
        Returns:
            4x4 transformation matrix from camera to end-effector
        """
        if len(self.ee_poses) < 3:
            self.get_logger().error('Need at least 3 calibration pairs')
            return None
        
        # This is a simplified version - in practice you would use
        # cv2.calibrateHandEye or a more robust solver
        
        self.get_logger().info(f'Computing calibration from {len(self.ee_poses)} pairs')
        
        # For now, return a placeholder
        # In real implementation, use proper hand-eye calibration algorithm
        T_ee_to_camera = np.eye(4)
        T_ee_to_camera[:3, 3] = [0.05, 0.0, 0.1]  # Example offset
        
        return T_ee_to_camera
    
    def transform_to_matrix(self, transform: TransformStamped) -> np.ndarray:
        """Convert TransformStamped to 4x4 matrix"""
        matrix = np.eye(4)
        
        t = transform.transform.translation
        matrix[0, 3] = t.x
        matrix[1, 3] = t.y
        matrix[2, 3] = t.z
        
        q = transform.transform.rotation
        rotation = R.from_quat([q.x, q.y, q.z, q.w])
        matrix[:3, :3] = rotation.as_matrix()
        
        return matrix


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = CoordinateTransformer()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()