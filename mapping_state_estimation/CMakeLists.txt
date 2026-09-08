cmake_minimum_required(VERSION 3.0.2)
project(mapping_state_estimation)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  tf2_ros
  tf2_geometry_msgs
  geometry_msgs
  nav_msgs
)

catkin_package(
  CATKIN_DEPENDS
    rospy
    sensor_msgs
    tf2_ros
    tf2_geometry_msgs
    geometry_msgs
    nav_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

# ============================================
# Python 节点安装
# ============================================
catkin_install_python(PROGRAMS
  scripts/imu_attitude_node.py
  scripts/cloud_preprocess_node.py
  scripts/lidar_odometry_node.py
  scripts/fastlio_bridge.py
  scripts/grid_map_node.py
  scripts/loop_closure_node.py
  scripts/robot_pose_publisher.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# ============================================
# Launch 文件和配置安装
# ============================================
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)
