# Start the robot arm

0. Source the environment
```bash
cd src
source install/setup.bash
```

1. Robot arm driver
```bash
ros2 launch kortex_bringup gen3_lite.launch.py robot_ip:=192.168.1.10 launch_rviz:=false
```

2. Motion planing by moveit
```bash
ros2 launch kinova_gen3_lite_moveit_config robot.launch.py robot_ip:=192.168.1.10
```

3. Launch the realsense camera  
```bash
ros2 launch realsense2_camera rs_launch.py enable_rgbd:=true enable_sync:=true align_depth.enable:=true
```

# Launch the project

1. Launch file  
```bash
ros2 launch kinova_graspnet_ros2 graspnet_kinova.launch.py
```

2. Start the yolo object detection  
Notice that you need to activate the conda env of graspnet
```bash
python scripts/yolo_detection_node.py
```

3. Start Detect!  
```
ros2 run kinova_graspnet_ros2 detect_grasps_client.py bottle 10
```
where you can change the object that you want to detect.  
args: 10, means present how many detected grasp pose.

4. Grasp  
Copy the output of the `3rd` terminal to a new terminal.