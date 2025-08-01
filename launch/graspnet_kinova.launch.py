#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    # Set up GraspNet environment
    graspnet_baseline_path = '/home/roar/graspnet/graspnet-baseline'
    
    # Environment variables for GraspNet
    set_pythonpath = SetEnvironmentVariable(
        name='PYTHONPATH',
        value=f'{graspnet_baseline_path}:{os.environ.get("PYTHONPATH", "")}'
    )
    
    # Declare launch arguments
    checkpoint_path_arg = DeclareLaunchArgument(
        'checkpoint_path',
        default_value='/home/roar/Downloads/checkpoint-rs.tar',
        description='Path to GraspNet model checkpoint'
    )
    
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time'
    )
    
    robot_ip_arg = DeclareLaunchArgument(
        'robot_ip',
        default_value='192.168.1.10',
        description='Kinova robot IP address'
    )
    
    camera_type_arg = DeclareLaunchArgument(
        'camera_type',
        default_value='realsense',
        description='Camera type (realsense, azure_kinect)'
    )
    
    # Get launch configurations
    checkpoint_path = LaunchConfiguration('checkpoint_path')
    use_sim_time = LaunchConfiguration('use_sim_time')
    robot_ip = LaunchConfiguration('robot_ip')
    camera_type = LaunchConfiguration('camera_type')
    
    # GraspNet detection service node
    grasp_detection_node = Node(
        package='kinova_graspnet_ros2',
        executable='grasp_detection_service.py',
        name='grasp_detection_service',
        output='screen',
        parameters=[
            PathJoinSubstitution([
                FindPackageShare('kinova_graspnet_ros2'),
                'config',
                'graspnet_params.yaml'
            ]),
            {
                'use_sim_time': use_sim_time,
                'checkpoint_path': checkpoint_path,
            }
        ]
    )
    
    # Kinova grasp controller node
    kinova_controller_node = Node(
        package='kinova_graspnet_ros2',
        executable='kinova_grasp_controller.py',
        name='kinova_grasp_controller',
        output='screen',
        parameters=[
            PathJoinSubstitution([
                FindPackageShare('kinova_graspnet_ros2'),
                'config',
                'graspnet_params.yaml'
            ]),
            {
                'use_sim_time': use_sim_time,
            }
        ]
    )
    
    # # Coordinate transformer node
    '''
    Find wheather this is wrong or not
    '''
    coordinate_transformer_node = Node(
        package='kinova_graspnet_ros2',
        executable='coordinate_transformer.py',
        name='coordinate_transformer',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'publish_static_transforms': False,
            # Example hand-eye calibration values - replace with actual calibration
            'camera_to_ee_translation': [0, 0.05639, -0.00305],
            'camera_to_ee_rotation': [0.0, 0.0, 0.0, 1.0]  # quaternion [x,y,z,w]
        }]
    )
    
    # Grasp visualizer node
    grasp_visualizer_node = Node(
        package='kinova_graspnet_ros2',
        executable='grasp_visualizer.py',
        name='grasp_visualizer',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time
        }]
    )
    
    # Grasp center frame publisher node
    grasp_center_publisher_node = Node(
        package='kinova_graspnet_ros2',
        executable='grasp_center_publisher.py',
        name='grasp_center_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'left_finger_frame': 'robotiq_85_left_finger_tip_link',
            'right_finger_frame': 'robotiq_85_right_finger_tip_link',
            'grasp_center_frame': 'grasp_center',
            'z_offset': 0.02,  # 3cm offset along Z axis
            'publish_rate': 50.0
        }]
    )
    
    # Real-time YOLO detection node
    '''
    DISABLED
    '''
    yolo_detection_node = Node(
        package='kinova_graspnet_ros2',
        executable='yolo_detection_node.py',
        name='yolo_detection_node',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'color_image_topic': '/camera/color/image_raw',
            'target_object_class': '',  # 检测所有对象，可以改为特定类别如 'bottle'
            'detection_fps': 2.0,  # 检测频率 (Hz)
            'confidence_threshold': 0.25
        }]
    )
    
    # Note: Camera frames and robot frames should be published by:
    # 1. Camera driver (camera_color_frame, camera_depth_frame)
    # 2. Robot driver (base_link, end_effector_link)
    # 3. Hand-eye calibration node (if needed)
    # The system will automatically get transforms from TF tree
    
    # Create launch description
    ld = LaunchDescription()
    
    # Add environment variables
    ld.add_action(set_pythonpath)
    
    # Add launch arguments
    ld.add_action(checkpoint_path_arg)
    ld.add_action(use_sim_time_arg)
    ld.add_action(robot_ip_arg)
    ld.add_action(camera_type_arg)
    
    # Add nodes
    ld.add_action(grasp_detection_node)
    ld.add_action(kinova_controller_node)
    ld.add_action(coordinate_transformer_node)
    ld.add_action(grasp_visualizer_node)
    ld.add_action(grasp_center_publisher_node)
    # ld.add_action(yolo_detection_node)
    
    return ld