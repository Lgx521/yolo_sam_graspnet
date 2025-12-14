cmake_minimum_required(VERSION 3.8)
project(kinova_graspnet_ros2)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(moveit_msgs REQUIRED)

# Generate service interfaces
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/DetectGrasps.srv"
  "srv/ExecuteGrasp.srv"
  "srv/BuildObstacles.srv"
  "srv/GenerateObstacles.srv"
  DEPENDENCIES geometry_msgs sensor_msgs std_msgs moveit_msgs
)

# Install Python modules (removed due to target conflict)
# ament_python_install_package(${PROJECT_NAME})

# Install Python scripts
install(PROGRAMS
  scripts/grasp_detection_service.py
  scripts/kinova_grasp_controller.py
  scripts/coordinate_transformer.py
  scripts/grasp_visualizer.py
  scripts/grasp_center_publisher.py
  scripts/grasp_visualizer_base_link.py
  scripts/detect_grasps_client.py
  scripts/yolo_detection_node.py
  scripts/obstacle_geometry_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

# Install config files
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}
)

# Install srv and msg files if any
install(DIRECTORY
  srv
  DESTINATION share/${PROJECT_NAME}
)

# Export dependencies for other packages
ament_export_dependencies(rosidl_default_runtime)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()