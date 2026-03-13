cmake_minimum_required(VERSION 3.0.2)
project(active_mapping)

# 强制 C++14 标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(catkin REQUIRED COMPONENTS
  roscpp rospy std_msgs sensor_msgs geometry_msgs nav_msgs
  cv_bridge pcl_ros pcl_conversions tf2_ros message_filters tf2_sensor_msgs
  quadrotor_msgs tf
)

find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)

catkin_package(
  CATKIN_DEPENDS roscpp rospy std_msgs sensor_msgs geometry_msgs nav_msgs cv_bridge pcl_ros pcl_conversions tf2_ros message_filters tf2_sensor_msgs quadrotor_msgs tf
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

add_executable(active_mapping_node 
  src/active_mapping_node.cpp
  src/map_builder_base.cpp
  src/cloud_map_builder.cpp
  src/surfel_map_builder.cpp
)
target_link_libraries(active_mapping_node 
  ${catkin_LIBRARIES} 
  ${PCL_LIBRARIES}
)

add_executable(heatmap_cluster_node
  src/heatmap_cluster_node.cpp
)
target_link_libraries(heatmap_cluster_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)

catkin_install_python(PROGRAMS
  scripts/airsim_frontend.py
  scripts/external_pose_tf_broadcaster.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
