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


# Start the yolo object detection
Notice that you need to activate the conda env of graspnet
```bash
python scripts/yolo_detection_node.py
```