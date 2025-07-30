#!/usr/bin/env python3

import numpy as np
import time
from typing import List, Tuple, Optional, Dict

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.duration import Duration
from rclpy.time import Time
import rclpy.executors

from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from sensor_msgs.msg import JointState, Image, CameraInfo
from std_srvs.srv import Trigger
from moveit_msgs.msg import Constraints, JointConstraint, PositionIKRequest
from moveit_msgs.action import MoveGroup
from moveit_msgs.srv import GetPositionIK
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import GripperCommand

import tf2_ros
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import tf2_geometry_msgs

from scipy.spatial.transform import Rotation as R

# Import custom service definitions
from kinova_graspnet_ros2.srv import DetectGrasps, ExecuteGrasp


class KinovaGraspController(Node):
    """ROS2 controller for executing grasps with Kinova Gen3 6DOF arm"""
    
    def __init__(self):
        super().__init__('kinova_grasp_controller')
        
        # Declare parameters
        self.declare_parameter('planning_group', 'manipulator')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('ee_frame', 'robotiq_85_base_link')
        self.declare_parameter('gripper_palm_frame', 'gripper_palm_center')
        self.declare_parameter('rgb_camera_frame', 'camera_color_frame')
        self.declare_parameter('depth_camera_frame', 'camera_depth_frame')
        self.declare_parameter('camera_frame', 'camera_depth_frame')
        self.declare_parameter('gripper_closed_position', 1.0)
        self.declare_parameter('gripper_open_position', 0.0)
        self.declare_parameter('approach_distance', -0.1)
        self.declare_parameter('retreat_distance', 0.3)
        self.declare_parameter('use_simplified_grasp', False)
        
        # Get parameters
        self.planning_group = self.get_parameter('planning_group').value
        self.base_frame = self.get_parameter('base_frame').value
        self.ee_frame = self.get_parameter('ee_frame').value
        self.gripper_palm_frame = self.get_parameter('gripper_palm_frame').value
        self.rgb_camera_frame = self.get_parameter('rgb_camera_frame').value
        self.depth_camera_frame = self.get_parameter('depth_camera_frame').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.gripper_closed_pos = self.get_parameter('gripper_closed_position').value
        self.gripper_open_pos = self.get_parameter('gripper_open_position').value
        self.approach_distance = self.get_parameter('approach_distance').value
        self.retreat_distance = self.get_parameter('retreat_distance').value
        self.use_simplified_grasp = self.get_parameter('use_simplified_grasp').value
        
        # Joint names for Kinova Gen3 6DOF
        self.joint_names = [
            'joint_1', 'joint_2', 'joint_3',
            'joint_4', 'joint_5', 'joint_6'
        ]
        self.gripper_joint_name = 'finger_joint'
        
        # Initialize TF2
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Initialize callback group for parallel execution
        self.callback_group = ReentrantCallbackGroup()
        
        # Create service clients
        self.grasp_detection_client = self.create_client(
            DetectGrasps,
            'detect_grasps',
            callback_group=self.callback_group
        )
        
        # MoveGroup action client
        self._move_group_client = ActionClient(
            self,
            MoveGroup,
            '/move_action',
            callback_group=self.callback_group
        )
        
        # IK service client
        self.ik_client = self.create_client(
            GetPositionIK,
            'compute_ik',
            callback_group=self.callback_group
        )
        
        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            10,
            callback_group=self.callback_group
        )
        
        # Gripper action client for Robotiq gripper
        self._gripper_action_client = ActionClient(
            self,
            GripperCommand,
            '/robotiq_gripper_controller/gripper_cmd',
            callback_group=self.callback_group
        )
        
        # Services
        self.execute_grasp_srv = self.create_service(
            ExecuteGrasp,
            'execute_grasp',
            self.execute_grasp_callback,
            callback_group=self.callback_group
        )
        
        self.auto_grasp_srv = self.create_service(
            Trigger,
            'auto_grasp',
            self.auto_grasp_callback,
            callback_group=self.callback_group
        )
        
        # Transform test service
        self.test_transform_srv = self.create_service(
            Trigger,
            'test_transforms',
            self.test_transforms_callback,
            callback_group=self.callback_group
        )
        
        # State variables
        self.current_joint_state = None
        
        # Wait for services
        self.get_logger().info('Waiting for required services...')
        self.wait_for_services()
        
        self.get_logger().info('Kinova grasp controller initialized')
    
    def wait_for_services(self):
        """Wait for required services to be available"""
        services_to_wait = [
            (self.grasp_detection_client, 'grasp detection'),
            (self.ik_client, 'IK service')
        ]
        
        for client, name in services_to_wait:
            if not client.wait_for_service(timeout_sec=10.0):
                self.get_logger().warn(f'{name} service not available')
        
        # Wait for MoveGroup action
        if not self._move_group_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().warn('MoveGroup action server not available')
        
        # Wait for Gripper action
        if not self._gripper_action_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().warn('Gripper action server not available')
    
    def joint_state_callback(self, msg: JointState):
        """Callback for joint state updates"""
        self.current_joint_state = msg
    
    #target_frame is "base_frame"
    def transform_pose(self, pose: PoseStamped, target_frame: str) -> Optional[PoseStamped]:
        """Transform pose to target frame"""
        try:
            self.get_logger().info(f'Transforming pose from {pose.header.frame_id} to {target_frame}')
            self.get_logger().info(f'Original pose: position=({pose.pose.position.x:.3f}, {pose.pose.position.y:.3f}, {pose.pose.position.z:.3f})')
            
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                pose.header.frame_id,
                Time(),
                timeout=Duration(seconds=1.0)
            )
            
            transformed_pose = tf2_geometry_msgs.do_transform_pose_stamped(pose, transform)
            
            self.get_logger().info(f'Transformed pose: position=({transformed_pose.pose.position.x:.3f}, {transformed_pose.pose.position.y:.3f}, {transformed_pose.pose.position.z:.3f})')
            
            return transformed_pose
            
        except TransformException as e:
            self.get_logger().error(f'Transform failed: {e}')
            return None
    
    def get_transform_matrix(self, target_frame: str, source_frame: str) -> Optional[np.ndarray]:
        """Get transformation matrix from source_frame to target_frame
        
        Usage example:
        # Get transform from camera to base: ros2 run tf2_ros tf2_echo base_link camera_depth_frame
        transform_matrix = self.get_transform_matrix('base_link', 'camera_depth_frame')
        """
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame, source_frame, 
                Time(), timeout=Duration(seconds=1.0)
            )
            
            # Extract translation
            trans = transform.transform.translation
            translation = np.array([trans.x, trans.y, trans.z])
            
            # Extract rotation (quaternion to rotation matrix)
            rot = transform.transform.rotation
            rotation_matrix = R.from_quat([rot.x, rot.y, rot.z, rot.w]).as_matrix()
            
            # Create 4x4 transformation matrix
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = rotation_matrix
            transform_matrix[:3, 3] = translation
            
            self.get_logger().info(f'Transform from {source_frame} to {target_frame}:')
            self.get_logger().info(f'Translation: {translation}')
            self.get_logger().info(f'Rotation matrix:\n{rotation_matrix}')
            
            return transform_matrix
            
        except TransformException as ex:
            self.get_logger().warn(f'Could not get transform from {source_frame} to {target_frame}: {ex}')
            return None
    
    def compute_ik(self, target_pose: PoseStamped) -> Optional[List[float]]:
        """Compute inverse kinematics for target pose"""
        if not self.current_joint_state:
            self.get_logger().error('No joint state available')
            return None
        
        # Create IK request
        ik_request = GetPositionIK.Request()
        ik_request.ik_request.group_name = self.planning_group
        ik_request.ik_request.robot_state.joint_state = self.current_joint_state
        ik_request.ik_request.pose_stamped = target_pose
        ik_request.ik_request.ik_link_name = self.ee_frame  # Use configured end effector frame
        ik_request.ik_request.timeout.sec = 5
        # Note: attempts attribute may not exist in this version of MoveIt
        
        # Call IK service
        future = self.ik_client.call_async(ik_request)
        
        # Wait for IK result without using spin_until_future_complete
        timeout = 10.0
        start_time = time.time()
        while not future.done() and (time.time() - start_time) < timeout:
            time.sleep(0.01)
        
        if not future.done() or future.result() is None:
            self.get_logger().error('IK service call failed or timed out')
            return None
        
        response = future.result()
        
        if response.error_code.val != response.error_code.SUCCESS:
            self.get_logger().error(f'IK failed with error code: {response.error_code.val}')
            self.get_logger().error(f'Target pose: position=({target_pose.pose.position.x:.3f}, {target_pose.pose.position.y:.3f}, {target_pose.pose.position.z:.3f})')
            self.get_logger().error(f'Target pose frame: {target_pose.header.frame_id}')
            return None
        
        # Extract joint values
        joint_values = []
        for joint_name in self.joint_names:
            if joint_name in response.solution.joint_state.name:
                idx = response.solution.joint_state.name.index(joint_name)
                joint_values.append(response.solution.joint_state.position[idx])
        
        return joint_values
    
    def move_to_joint_positions(self, joint_positions: List[float],
                               max_velocity_scaling: float = 0.3,
                               max_acceleration_scaling: float = 0.3) -> bool:
        """Move arm to joint positions using MoveGroup"""
        if len(joint_positions) != len(self.joint_names):
            self.get_logger().error(f'Expected {len(self.joint_names)} joint positions, got {len(joint_positions)}')
            return False
        
        # Create MoveGroup goal
        goal_msg = MoveGroup.Goal()
        goal_msg.request.group_name = self.planning_group
        goal_msg.request.num_planning_attempts = 5
        goal_msg.request.allowed_planning_time = 5.0
        goal_msg.request.max_velocity_scaling_factor = max_velocity_scaling
        goal_msg.request.max_acceleration_scaling_factor = max_acceleration_scaling
        
        # Create joint constraints
        constraints = Constraints()
        for joint_name, position in zip(self.joint_names, joint_positions):
            joint_constraint = JointConstraint()
            joint_constraint.joint_name = joint_name
            joint_constraint.position = float(position)
            joint_constraint.tolerance_above = 0.001
            joint_constraint.tolerance_below = 0.001
            joint_constraint.weight = 1.0
            constraints.joint_constraints.append(joint_constraint)
        
        goal_msg.request.goal_constraints = [constraints]
        
        # Send goal
        self.get_logger().info('Sending movement goal to MoveGroup')
        goal_future = self._move_group_client.send_goal_async(goal_msg)
        
        # Wait for goal acceptance
        timeout = 30.0  # Increased timeout for slower movements
        start_time = time.time()
        while not goal_future.done() and (time.time() - start_time) < timeout:
            time.sleep(0.01)
        
        if not goal_future.done():
            self.get_logger().error('Movement goal send timeout')
            return False
            
        goal_handle = goal_future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Movement goal rejected')
            return False
        
        # Wait for result
        result_future = goal_handle.get_result_async()
        start_time = time.time()
        while not result_future.done() and (time.time() - start_time) < timeout:
            time.sleep(0.01)
        
        if not result_future.done():
            self.get_logger().error('Movement result timeout')
            return False
            
        result = result_future.result()
        if result is None:
            self.get_logger().error('Movement result is None')
            return False
            
        if result.result.error_code.val == 1:  # SUCCESS
            self.get_logger().info('Movement completed successfully')
            return True
        else:
            self.get_logger().error(f'Movement failed with error code: {result.result.error_code.val}')
            return False
    
    def move_to_pose(self, target_pose: PoseStamped,
                     max_velocity_scaling: float = 0.3,
                     max_acceleration_scaling: float = 0.3) -> bool:
        """Move end effector to target pose"""
        # Compute IK
        joint_positions = self.compute_ik(target_pose)
        if joint_positions is None:
            return False
        
        # Execute movement
        return self.move_to_joint_positions(
            joint_positions,
            max_velocity_scaling,
            max_acceleration_scaling
        )
    
    def control_gripper(self, position: float, max_effort: float = 100.0) -> bool:
        """Control gripper position using Robotiq action server
        
        Args:
            position: Target position (0.0 = fully open, 1.0 = fully closed)
            max_effort: Maximum effort/force to apply
        """
        try:
            # Create goal for gripper action - matching the command line format exactly
            goal_msg = GripperCommand.Goal()
            goal_msg.command.position = float(position)  # Ensure it's a float
            goal_msg.command.max_effort = float(max_effort)
            
            self.get_logger().info(f'Sending gripper command: position={position:.3f}, max_effort={max_effort:.1f}')
            
            # Send goal asynchronously
            send_goal_future = self._gripper_action_client.send_goal_async(
                goal_msg,
                feedback_callback=lambda feedback: self.get_logger().debug(f'Gripper feedback: {feedback}')
            )
            
            # Use the executor to wait for the future
            executor = rclpy.get_global_executor()
            if executor:
                # Add a small timeout to process callbacks
                timeout_sec = 5.0
                start_time = self.get_clock().now()
                while not send_goal_future.done():
                    executor.spin_once(timeout_sec=0.1)
                    if (self.get_clock().now() - start_time).nanoseconds / 1e9 > timeout_sec:
                        self.get_logger().error('Timeout waiting for gripper goal acceptance')
                        return False
            else:
                # Fallback to simple wait
                timeout = 5.0
                start_time = time.time()
                while not send_goal_future.done() and (time.time() - start_time) < timeout:
                    time.sleep(0.01)
            
            goal_handle = send_goal_future.result()
            if not goal_handle.accepted:
                self.get_logger().error('Gripper goal was rejected')
                return False
            
            self.get_logger().info('Gripper goal accepted')
            
            # Get the result
            get_result_future = goal_handle.get_result_async()
            
            # Wait for result with timeout
            if executor:
                start_time = self.get_clock().now()
                while not get_result_future.done():
                    executor.spin_once(timeout_sec=0.1)
                    if (self.get_clock().now() - start_time).nanoseconds / 1e9 > timeout_sec:
                        self.get_logger().warn('Timeout waiting for gripper result, but goal was accepted')
                        return True  # Assume success since goal was accepted
            else:
                # Fallback to simple wait
                timeout = 5.0
                start_time = time.time()
                while not get_result_future.done() and (time.time() - start_time) < timeout:
                    time.sleep(0.01)
            
            if get_result_future.done():
                result = get_result_future.result().result
                self.get_logger().info(f'Gripper action completed')
                return True
            else:
                self.get_logger().warn('Gripper result not received, but assuming success')
                return True
                
        except Exception as e:
            self.get_logger().error(f'Gripper control error: {str(e)}')
            import traceback
            self.get_logger().error(f'Traceback: {traceback.format_exc()}')
            return False
    
    def compute_approach_pose(self, grasp_pose: PoseStamped, distance: float) -> PoseStamped:
        """Compute approach pose along grasp approach vector"""
        
        approach_pose = PoseStamped()
        approach_pose.header = grasp_pose.header
        
        # Extract rotation from grasp pose
        q = grasp_pose.pose.orientation
        rotation = R.from_quat([q.x, q.y, q.z, q.w])
        rotation_matrix = rotation.as_matrix()
        
        # Approach vector is along positive Z axis of grasp frame
        # In TF camera frame, Z-axis points forward (approach direction)
        approach_vector = rotation_matrix[:, 2]
        
        # Compute approach position
        approach_pose.pose.position.x = grasp_pose.pose.position.x + approach_vector[0] * distance
        approach_pose.pose.position.y = grasp_pose.pose.position.y + approach_vector[1] * distance
        approach_pose.pose.position.z = grasp_pose.pose.position.z + approach_vector[2] * distance
        
        # Keep same orientation
        approach_pose.pose.orientation = grasp_pose.pose.orientation
        
        return approach_pose
    
    def transform_grasp_center_to_ee(self, grasp_pose_camera: PoseStamped) -> Optional[PoseStamped]:
        """
        Transforms a grasp pose, defined for the 'grasp_center' frame in the camera's view,
        into a target pose for the robot's end-effector ('ee_frame') in the 'base_link' frame.

        Args:
            grasp_pose_camera: The desired pose for 'grasp_center', relative to the camera frame.

        Returns:
            The corresponding target pose for 'ee_frame', relative to 'base_link', or None on failure.
        """
        try:
            # Step 1: Transform the desired grasp_center pose from the camera frame to the base frame.
            # This gives us T_base_gc_target.
            grasp_center_pose_base = self.transform_pose(grasp_pose_camera, self.base_frame)
            if grasp_center_pose_base is None:
                self.get_logger().error("Failed to transform grasp center pose to base frame.")
                return None
            
            T_base_gc_target = self.pose_stamped_to_matrix(grasp_center_pose_base)

            # Step 2: Get the static transform from 'ee_frame' to 'grasp_center'.
            # Note: lookup_transform('target', 'source') gives T_target_source.
            # We want T_gc_ee, but TF gives us T_ee_gc. So we will invert it.
            try:
                # Let's get T_ee_gc (transform from grasp_center to ee_frame)
                transform_ee_to_gc = self.tf_buffer.lookup_transform(
                    self.ee_frame,       # Target Frame
                    'grasp_center',      # Source Frame
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=1.0)
                )
                T_ee_gc_static = self.transform_to_matrix(transform_ee_to_gc)

            except TransformException as e:
                self.get_logger().error(f"Failed to look up transform from 'grasp_center' to '{self.ee_frame}': {e}")
                self.get_logger().error("Ensure grasp_center_publisher.py is running and publishing the TF.")
                return None

            # Step 3: Calculate the target pose for the end-effector.
            # Formula: T_base_ee_target = T_base_gc_target * (T_ee_gc_static)^-1
            # (T_ee_gc_static)^-1 is the transform from grasp_center to ee_frame, T_gc_ee_static.
            T_gc_ee_static = np.linalg.inv(T_ee_gc_static)
            
            # The multiplication order is crucial.
            T_base_ee_target = T_base_gc_target @ T_gc_ee_static
            
            # Step 4: Convert the resulting matrix back to a PoseStamped message.
            ee_pose_base = self.matrix_to_pose_stamped(T_base_ee_target, self.base_frame)

            # Logging for verification
            self.get_logger().info(f'Grasp transformation successful:')
            self.get_logger().info(f'  Input Grasp Center (camera): pos({grasp_pose_camera.pose.position.x:.3f}, {grasp_pose_camera.pose.position.y:.3f}, {grasp_pose_camera.pose.position.z:.3f})')
            self.get_logger().info(f'  Target Grasp Center (base): pos({grasp_center_pose_base.pose.position.x:.3f}, {grasp_center_pose_base.pose.position.y:.3f}, {grasp_center_pose_base.pose.position.z:.3f})')
            self.get_logger().info(f'  Calculated EE Target (base): pos({ee_pose_base.pose.position.x:.3f}, {ee_pose_base.pose.position.y:.3f}, {ee_pose_base.pose.position.z:.3f})')
            
            return ee_pose_base

        except Exception as e:
            self.get_logger().error(f'An unexpected error occurred in transform_grasp_center_to_ee: {str(e)}')
            import traceback
            self.get_logger().error(f'Traceback: {traceback.format_exc()}')
            return None
    
    def transform_to_matrix(self, transform: tf2_ros.TransformStamped) -> np.ndarray:
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
    
    def pose_stamped_to_matrix(self, pose: PoseStamped) -> np.ndarray:
        """Convert PoseStamped to 4x4 transformation matrix"""
        matrix = np.eye(4)
        
        # Translation
        matrix[0, 3] = pose.pose.position.x
        matrix[1, 3] = pose.pose.position.y
        matrix[2, 3] = pose.pose.position.z
        
        # Rotation
        q = pose.pose.orientation
        rotation = R.from_quat([q.x, q.y, q.z, q.w])
        matrix[:3, :3] = rotation.as_matrix()
        
        return matrix
    
    def matrix_to_pose_stamped(self, matrix: np.ndarray, frame_id: str) -> PoseStamped:
        """Convert 4x4 transformation matrix to PoseStamped"""
        pose = PoseStamped()
        pose.header.frame_id = frame_id
        pose.header.stamp = self.get_clock().now().to_msg()
        
        # Translation
        pose.pose.position.x = float(matrix[0, 3])
        pose.pose.position.y = float(matrix[1, 3])
        pose.pose.position.z = float(matrix[2, 3])
        
        # Rotation
        rotation = R.from_matrix(matrix[:3, :3])
        quat = rotation.as_quat()  # [x, y, z, w]
        pose.pose.orientation.x = float(quat[0])
        pose.pose.orientation.y = float(quat[1])
        pose.pose.orientation.z = float(quat[2])
        pose.pose.orientation.w = float(quat[3])
        
        return pose
    
    def execute_grasp_sequence(self, grasp_pose: PoseStamped, grasp_width: float,
                              approach_distance: float,
                              max_velocity_scaling: float = 0.3,
                              max_acceleration_scaling: float = 0.3) -> Tuple[bool, str]:
        """Execute simplified grasp sequence"""
        try:
            # Transform grasp pose from camera frame to end-effector pose in base frame
            # This handles the conversion from grasp_center to end-effector
            ee_pose_base = self.transform_grasp_center_to_ee(grasp_pose)
            if ee_pose_base is None:
                return False, "Failed to transform grasp pose to end-effector frame"
            
            # 1. Open gripper
            self.get_logger().info('Opening gripper')
            self.control_gripper(self.gripper_open_pos)
            
            # 2. Move directly to grasp pose (simplified - no approach phase)
            self.get_logger().info('Moving directly to grasp pose')
            if not self.move_to_pose(ee_pose_base, max_velocity_scaling * 0.5, max_acceleration_scaling * 0.5):
                return False, "Failed to reach grasp pose"
            
            # 3. Close gripper
            self.get_logger().info(f'Closing gripper to width: {grasp_width}')
            # Convert grasp width to gripper position
            # grasp_width: predicted opening width (can exceed 0.085m)
            # gripper_position: 0.0 = fully open, 1.0 = fully closed
            
            # If predicted width is larger than max opening, use a reasonable closing position
            if grasp_width > 0.085:
                # For very wide objects, close to about 80% to ensure good grip
                gripper_position = 0.8
                self.get_logger().info(f'Predicted width {grasp_width:.3f}m exceeds max opening 0.085m, using position {gripper_position:.3f}')
            else:
                # Normal case: map grasp width to gripper position
                # Small width = more closed, large width = more open
                normalized_width = grasp_width / 0.085  # 0 to 1
                gripper_position = 1.0 - normalized_width  # Invert: small width = more closed
                gripper_position = max(min(gripper_position, 1.0), 0.1)  # Clamp to 0.1-1.0 (never fully open)
                self.get_logger().info(f'Gripper position calculated: {gripper_position:.3f} (0=open, 1=closed) for width {grasp_width:.3f}m')
            
            self.control_gripper(gripper_position)
            
            return True, "Simplified grasp executed successfully"
            
        except Exception as e:
            self.get_logger().error(f'Grasp execution failed: {str(e)}')
            import traceback
            self.get_logger().error(f'Full traceback: {traceback.format_exc()}')
            return False, f"Exception during grasp execution: {str(e)}"
    
    def execute_grasp_sequence_with_approach(self, grasp_pose: PoseStamped, grasp_width: float,
                              approach_distance: float,
                              max_velocity_scaling: float = 0.3,
                              max_acceleration_scaling: float = 0.3) -> Tuple[bool, str]:
        """Execute complete grasp sequence with approach and retreat"""
        try:
            # Log grasp parameters
            self.get_logger().info(f'Grasp parameters: width={grasp_width:.3f}m, approach_dist={approach_distance:.3f}m')
            # Transform grasp pose from camera frame to end-effector pose in base frame
            # This handles the conversion from grasp_center to end-effector
            ee_pose_base = self.transform_grasp_center_to_ee(grasp_pose)
            if ee_pose_base is None:
                return False, "Failed to transform grasp pose to end-effector frame"
            
            # 1. Open gripper
            self.get_logger().info('Opening gripper')
            self.control_gripper(self.gripper_open_pos)
            
            # 2. Move to approach pose 
            approach_pose = self.compute_approach_pose(ee_pose_base, approach_distance)
            
            if not self.move_to_pose(approach_pose, max_velocity_scaling, max_acceleration_scaling):
                return False, "Failed to reach approach pose"
            
            # 3. Move to grasp pose
            self.get_logger().info('Moving to grasp pose')
            if not self.move_to_pose(ee_pose_base, max_velocity_scaling * 0.5, max_acceleration_scaling * 0.5):
                return False, "Failed to reach grasp pose"
            
            # 4. Close gripper
            self.get_logger().info(f'Closing gripper to width: {grasp_width}')
            # Convert grasp width to gripper position
            # grasp_width: predicted opening width (can exceed 0.085m)
            # gripper_position: 0.0 = fully open, 1.0 = fully closed
            
            # If predicted width is larger than max opening, use a reasonable closing position
            if grasp_width > 0.085:
                # For very wide objects, close to about 80% to ensure good grip
                gripper_position = 0.8
                self.get_logger().info(f'Predicted width {grasp_width:.3f}m exceeds max opening 0.085m, using position {gripper_position:.3f}')
            else:
                # Normal case: map grasp width to gripper position
                # Small width = more closed, large width = more open
                normalized_width = grasp_width / 0.085  # 0 to 1
                gripper_position = 1.0 - normalized_width  # Invert: small width = more closed
                gripper_position = max(min(gripper_position, 1.0), 0.1)  # Clamp to 0.1-1.0 (never fully open)
                self.get_logger().info(f'Gripper position calculated: {gripper_position:.3f} (0=open, 1=closed) for width {grasp_width:.3f}m')
            
            self.control_gripper(gripper_position)
            
            # 5. Retreat (move upward regardless of retreat_distance sign)
            # Always retreat upward by the absolute value of retreat_distance
            retreat_distance_abs = abs(self.retreat_distance)
            retreat_pose = PoseStamped()
            retreat_pose.header = ee_pose_base.header
            retreat_pose.pose = ee_pose_base.pose
            # Move straight up in base frame
            retreat_pose.pose.position.z += retreat_distance_abs
            
            self.get_logger().info(f'Retreating upward by {retreat_distance_abs:.3f}m')
            if not self.move_to_pose(retreat_pose, max_velocity_scaling, max_acceleration_scaling):
                self.get_logger().warn('Failed to retreat, but grasp may still be successful')
            
            return True, "Grasp executed successfully"
            
        except Exception as e:
            self.get_logger().error(f'Grasp execution failed: {str(e)}')
            import traceback
            self.get_logger().error(f'Full traceback: {traceback.format_exc()}')
            return False, f"Exception during grasp execution: {str(e)}"
    
    def execute_grasp_callback(self, request: ExecuteGrasp.Request, response: ExecuteGrasp.Response):
        """Service callback for executing a grasp"""
        self.get_logger().info('Received grasp execution request')
        
        start_time = time.time()
        
        # Choose execution method based on parameter
        if self.use_simplified_grasp:
            self.get_logger().info('Using simplified grasp execution (direct movement)')
            success, message = self.execute_grasp_sequence(
                request.grasp_pose,
                request.grasp_width,
                request.approach_distance,
                request.max_velocity_scaling,
                request.max_acceleration_scaling
            )
        else:
            self.get_logger().info('Using full grasp execution (with approach and retreat)')
            success, message = self.execute_grasp_sequence_with_approach(
                request.grasp_pose,
                request.grasp_width,
                request.approach_distance,
                request.max_velocity_scaling,
                request.max_acceleration_scaling
            )
        
        response.success = success
        response.message = message
        response.execution_time = time.time() - start_time
        
        # Get final pose
        if self.current_joint_state:
            # Could compute forward kinematics here to get final pose
            # For now, just copy the requested pose
            response.final_pose = request.grasp_pose
        
        return response
    
    def auto_grasp_callback(self, request: Trigger.Request, response: Trigger.Response):
        """Service callback for automatic grasp detection and execution"""
        self.get_logger().info('Auto grasp requested - this would trigger camera capture and grasp detection')
        
        # This is a placeholder - in a real implementation, you would:
        # 1. Capture current camera images
        # 2. Call grasp detection service
        # 3. Select best grasp
        # 4. Execute grasp
        
        response.success = True
        response.message = "Auto grasp functionality not fully implemented yet"
        
        return response
    
    def test_transforms_callback(self, request: Trigger.Request, response: Trigger.Response):
        """Service callback to test coordinate transforms"""
        self.get_logger().info('Testing coordinate transforms...')
        
        transform_tests = [
            (self.base_frame, self.depth_camera_frame, "Camera depth to base"),
            (self.base_frame, self.rgb_camera_frame, "Camera RGB to base"), 
            (self.depth_camera_frame, self.rgb_camera_frame, "RGB to depth camera"),
            (self.base_frame, self.ee_frame, "Gripper mount to base"),
            (self.base_frame, self.gripper_palm_frame, "Gripper palm to base"),
            (self.ee_frame, self.gripper_palm_frame, "Gripper palm to mount"),
            (self.gripper_palm_frame, self.depth_camera_frame, "Camera depth to gripper palm"),
            (self.gripper_palm_frame, self.rgb_camera_frame, "Camera RGB to gripper palm")
        ]
        
        success_count = 0
        total_tests = len(transform_tests)
        messages = []
        
        try:
            for target_frame, source_frame, description in transform_tests:
                self.get_logger().info(f'=== Testing {description} ===')
                transform_matrix = self.get_transform_matrix(target_frame, source_frame)
                
                if transform_matrix is not None:
                    success_count += 1
                    messages.append(f"✓ {description}: {source_frame} -> {target_frame}")
                else:
                    messages.append(f"✗ {description}: Failed to get transform {source_frame} -> {target_frame}")
            
            response.success = success_count == total_tests
            response.message = f"Transform test results ({success_count}/{total_tests} successful):\n" + "\n".join(messages)
            
            if success_count == total_tests:
                self.get_logger().info("All coordinate transforms are available!")
            else:
                self.get_logger().warn(f"Only {success_count}/{total_tests} transforms are available")
                
        except Exception as e:
            response.success = False
            response.message = f"Error testing transforms: {str(e)}"
            self.get_logger().error(f"Transform test failed: {e}")
        
        return response


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = KinovaGraspController()
        
        # Use MultiThreadedExecutor for handling callbacks
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        executor.spin()
        
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()