# Kinova GraspNet ROS2 Integration

This package provides ROS2 integration between GraspNet grasp detection and Kinova Gen3 6DOF robotic arm, enhanced with intelligent object segmentation using YOLO-World and SAM, and automatic camera data acquisition.

## Overview

The system consists of several ROS2 nodes that work together:

1. **Grasp Detection Service** - Automatically subscribes to camera topics and processes RGBD images using GraspNet to detect grasp poses
2. **Kinova Grasp Controller** - Executes detected grasps using MoveIt2 motion planning with automatic coordinate transformation
3. **Coordinate Transformer** - Handles transformations between camera and robot frames via TF2
4. **Grasp Visualizer** - Provides RViz visualization of detected grasps
5.  **Obstacle Geometry Node** - Models the geometry of obstacles (all non-target objects) using Alpha Shapes for collision-aware motion planning.

## Key Features

- **Automatic Camera Data Acquisition**: Subscribes to camera topics automatically, no manual image passing required
- **Intelligent Object Segmentation**: Uses YOLO-World + SAM for precise object detection and segmentation
- **Dynamic Coordinate Transformation**: Automatically gets transforms from TF tree (no hardcoded transforms)
- **Configurable Target Objects**: Specify which objects to grasp via service parameters
- **Dual Camera Support**: Separate intrinsics for RGB and depth cameras
- **Complete Grasp Pipeline**: From detection to execution in two simple service calls
- **Advanced Obstacle Modeling**: Generates non-convex Alpha Shapes of obstacles for tighter, more accurate collision models, enabling planning in cluttered environments.

## Prerequisites

- ROS2 Humble or newer
- Python 3.8+
- PyTorch with CUDA support
- Open3D
- MoveIt2
- Kinova ROS2 packages (`ros2_kortex`)
- Camera drivers (RealSense, Azure Kinect, etc.)
- GraspNet dependencies
- Ultralytics (for YOLO-World and SAM integration)

## Installation

1. Clone this package into your ROS2 workspace:
```bash
cd ~/ros2_ws/src
ln -s /home/q/Graspnet/graspnet-baseline/kinova_graspnet_ros2 .
```

2. Install Python dependencies:
```bash
pip install torch torchvision open3d scipy opencv-python ultralytics
Please follow the instruction in Origin Grasppnet first.
```

3. Build the workspace:
```bash
cd /home/q/Graspnet/graspnet-baseline/kinova_graspnet_ros2/
colcon build --packages-select kinova_graspnet_ros2
source install/setup.bash
```

4. Download GraspNet pretrained models:
```bash
# For RealSense camera
wget https://graspnet.net/checkpoint-rs.tar -P ~/Downloads/
```

## Configuration

Edit `config/graspnet_params.yaml` to configure:

### Camera Configuration
```yaml
# RGB Camera parameters
rgb_camera_intrinsics:
  fx: 1386.11  # Adjust for your camera
  fy: 1386.11
  cx: 959.5
  cy: 539.5
  width: 1920
  height: 1080

# Depth Camera parameters  
depth_camera_intrinsics:
  fx: 606.438  # Adjust for your camera
  fy: 606.351
  cx: 637.294
  cy: 366.992
  width: 1280
  height: 720

# Camera topics
color_image_topic: "/camera/color/image_raw"
depth_image_topic: "/camera/depth/image_raw"
camera_info_topic: "/camera/color/camera_info"
```

### Frame Configuration
```yaml
# Frame names (adjust to match your setup)
rgb_camera_frame: "camera_color_frame"
depth_camera_frame: "camera_depth_frame"  
base_frame: "base_link"
```

### Target Object
```yaml
target_object_class: "bottle"  # Default object to grasp
```

## Complete Usage Pipeline

### Step 1: Start Robot and Camera Drivers
```bash
# Terminal 1: Start Kinova robot driver with MoveIt
ros2 launch kinova_gen3_6dof_robotiq_2f_85_moveit_config robot.launch.py robot_ip:=192.168.1.10

# Terminal 2: Start Kinova vision driver (RGB-D camera)
ros2 launch kinova_vision kinova_vision.launch.py depth_registration:=true
```

### Step 2: Launch GraspNet System
```bash
# Terminal 3: Start GraspNet nodes
source install/setup.bash

ros2 launch kinova_graspnet_ros2 graspnet_kinova.launch.py 
```

The system will automatically:
- Subscribe to camera topics
- Load GraspNet model
- Wait for camera data to be ready
- Start services for grasp detection and execution

### Step 3: Execute Grasp Detection

#### 3.1: Detect Grasp Poses
First, run the grasp detection client to find potential grasps for your target object. This step also visualizes the grasps in RViz.  
```bash
# Detect grasps for a specific object (e.g., bottle)
ros2 run kinova_graspnet_ros2 detect_grasps_client.py bottle 10

# Or detect all graspable objects
ros2 run kinova_graspnet_ros2 detect_grasps_client.py "" 10
```

The client will display formatted results and provide ready-to-use execution commands.  
Copy your desired grasp command, but do not execute it yet.


#### 3.2: Generate and Visualize Obstacles
Next, call the obstacle generation service. This service will:  
1. Identify all objects in the scene.
2. Subtract the target object (e.g., 'apple').
3. Model the remaining objects (obstacles) as tight-fitting Alpha Shapes.
4. Publish the obstacle models to RViz for visualization.

Run the obstacle geometry node  

```bash
ros2 run kinova_graspnet_ros2 obstacle_geometry_node.py
```

```bash
# Call the service, specifying the same target object to exclude it from obstacles
ros2 service call /generate_obstacles kinova_graspnet_ros2/srv/GenerateObstacles "{target_object_class: 'apple'}"
```

Now, in RViz, you should see purple, form-fitting meshes around all obstacles.





### Step 4: Execute Grasp
```bash
# Execute a specific grasp (use pose from detection response)
ros2 service call /execute_grasp kinova_graspnet_ros2/srv/ExecuteGrasp \
    "{grasp_pose: <pose_from_detection>, grasp_width: 0.05, approach_distance: 0.1}"
```

## Service Interfaces

### Grasp Detection Service (`/detect_grasps`)
**Input:**
- `target_object_class` (string): Object to grasp (e.g., "bottle", "cup", "apple")
- `max_grasps` (int): Maximum number of grasps to return
- `workspace_min/max` (Point): Optional workspace constraints

**Output:**
- `grasp_poses` (PoseStamped[]): Detected grasp poses in camera frame
- `grasp_scores` (float[]): Quality scores for each grasp
- `grasp_widths` (float[]): Required gripper widths
- `success` (bool): Detection success status

### Grasp Execution Service (`/execute_grasp`)
**Input:**
- `grasp_pose` (PoseStamped): Target grasp pose
- `grasp_width` (float): Gripper width for grasping
- `approach_distance` (float): Distance to approach from
- `max_velocity_scaling` (float): Motion speed scaling

**Output:**
- `success` (bool): Execution success status
- `execution_time` (float): Time taken for execution
- `final_pose` (PoseStamped): Final robot pose

### Obstacle Generation Service (`/generate_obstacles`)
**Input:**
- `target_object_class` (string): The object to be grasped, which will be excluded from the obstacle set.

**Output:**
- `success` (bool): Generation success status.
- `message` (string): A descriptive message about the result.

### Additional Services
- `/test_transforms`: Test coordinate transformations
- `/auto_grasp`: Placeholder for full automation (not implemented)

## Coordinate Systems

The system automatically handles coordinate transformations via TF2:

1. **Camera frames**: `camera_color_frame`, `camera_depth_frame`
2. **Robot frame**: `base_link` 
3. **Automatic transformation**: Camera poses → Robot base frame

Use this command to verify transforms:
```bash
ros2 run tf2_ros tf2_echo base_link camera_depth_frame
ros2 service call /test_transforms std_srvs/srv/Trigger
```

## Topics

### Published Topics
- `/grasp_markers` (visualization_msgs/MarkerArray): Grasp visualizations for RViz
- `/grasp_point_cloud` (sensor_msgs/PointCloud2): Processed point cloud
- `/yolo_detection_visualization` (sensor_msgs/Image): YOLO detection results with bounding boxes, labels, and confidence scores
- `/obstacle_markers` (visualization_msgs/MarkerArray): Visualizations of obstacle geometry (Alpha Shapes) for RViz.

### Subscribed Topics  
- `/camera/color/image_raw` (sensor_msgs/Image): RGB images
- `/camera/depth/image_raw` (sensor_msgs/Image): Depth images
- `/camera/color/camera_info` (sensor_msgs/CameraInfo): Camera intrinsics
- `/joint_states` (sensor_msgs/JointState): Robot joint states

## Troubleshooting

### Common Issues

1. **"Camera data not ready"**
   - Check camera drivers are running: `ros2 topic list | grep camera`
   - Verify topic names in config file match published topics

2. **"Transform failed"**
   - Ensure robot and camera drivers publish TF frames
   - Check frame names: `ros2 run tf2_tools view_frames`
   - Test transforms: `ros2 service call /test_transforms std_srvs/srv/Trigger`

3. **"No grasps detected"**
   - Check camera exposure and lighting
   - Verify object is in workspace limits
   - Try different target object classes

4. **"CUDA out of memory"**
   - Reduce `num_point` parameter in config file
   - Use CPU: set `device: "cpu"` in config

5. **Motion planning failures**
   - Verify MoveIt is running: `ros2 topic list | grep move_group`
   - Check robot URDF and collision geometry
   - Ensure robot is not in collision
6. **Obstacles are not detected or are inaccurate**
   - The algorithm relies on a "point cloud subtraction" method. Ensure the `target_object_class` is segmented correctly.
   - If obstacles are too small or too large, adjust the `min_bound`/`max_bound` workspace parameters and `voxel_size` in the `obstacle_geometry_node.py` script.
   - If the obstacle shape is too "blobby" or fragmented, fine-tune the `alpha_value` in the same script.

### Visualizing in RViz2
To view the YOLO detection results in RViz2:
1. Add an Image display
2. Set the topic to `/yolo_detection_visualization`
3. The image will show bounding boxes, object classes, and confidence scores for detected objects

To view the obstacle models in RViz2:
1. Add a MarkerArray display.
2. Set the topic to `/obstacle_markers`.
3. The obstacles will appear as semi-transparent purple meshes.

**实时检测**: 启动launch文件后，系统会自动以2Hz的频率进行YOLO检测并发布可视化结果，无需额外调用服务。

## Advanced Configuration

### Different Camera Types
```bash
# For Azure Kinect
ros2 launch kinova_graspnet_ros2 graspnet_kinova.launch.py \
    checkpoint_path:=/path/to/checkpoint-kn.tar \
    camera_type:=azure_kinect

# For custom camera (modify topic names in config)
```

### Custom Object Classes
Supported YOLO-World object classes include:
- Daily objects: `bottle`, `cup`, `bowl`, `apple`, `banana`
- Tools: `scissors`, `knife`, `hammer` 
- Electronics: `cell phone`, `laptop`, `mouse`

### Hand-Eye Calibration
The system automatically uses TF transforms. For manual calibration:
1. Use standard ROS hand-eye calibration tools
2. Publish static transform between camera and robot
3. System will automatically use the transform

## Integration with Existing Systems

This package integrates seamlessly with:
- **MoveIt2**: Motion planning and execution
- **RealSense/Azure Kinect**: Camera drivers  
- **Kinova `ros2_kortex`**: Robot drivers
- **Custom perception pipelines**: Via configurable topics

## Performance Notes

- **Detection time**: ~2-3 seconds for 256 grasp candidates
- **Execution time**: ~10-15 seconds for complete grasp sequence
- **Memory usage**: ~4GB GPU memory for full model
- **Accuracy**: Depends on lighting, object type, and camera quality

## Citation

If you use this code, please cite:
- [GraspNet-1Billion paper](https://graspnet.net/)
- [Kinova ROS2 packages](https://github.com/Kinovarobotics/ros2_kortex)
- [Ultralytics YOLO](https://ultralytics.com/)