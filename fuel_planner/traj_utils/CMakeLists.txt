cmake_minimum_required(VERSION 2.8.3)
project(traj_utils)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall")

find_package(catkin REQUIRED COMPONENTS
  bspline
  bspline_opt
  path_searching
  poly_traj
  roscpp
  std_msgs
  cv_bridge
  message_generation
)

find_package(Eigen3 REQUIRED)
find_package(PCL 1.7 REQUIRED)

add_message_files(
  FILES
  PolyTraj.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
 INCLUDE_DIRS include
 LIBRARIES traj_utils
 CATKIN_DEPENDS path_searching bspline bspline_opt poly_traj message_runtime
#  DEPENDS system_lib
)

include_directories( 
    SYSTEM 
    include 
    ${catkin_INCLUDE_DIRS}
    ${Eigen3_INCLUDE_DIRS} 
    ${PCL_INCLUDE_DIRS}
)

link_directories(${PCL_LIBRARY_DIRS})

add_library( traj_utils
    src/planning_visualization.cpp 
    )
target_link_libraries( traj_utils
    ${catkin_LIBRARIES} 
    )  
add_dependencies(traj_utils ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

add_executable(process_msg 
    src/process_msg.cpp
)
target_link_libraries( process_msg
    ${catkin_LIBRARIES} 
    ${PCL_LIBRARIES}
    )  
add_dependencies(process_msg ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
