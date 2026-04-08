cmake_minimum_required(VERSION 3.0.2)
project(gpu_visibility_checker)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -fPIC")

find_package(catkin REQUIRED)
find_package(Eigen3 REQUIRED)

catkin_package(
    INCLUDE_DIRS include
    LIBRARIES gpu_visibility_checker
    DEPENDS EIGEN3
)

include_directories(
    include
    ${EIGEN3_INCLUDE_DIR}
)

# 只编译 spherical_binning (纯 C++ 代码，不需要 SYCL)
add_library(gpu_visibility_checker
    src/spherical_binning.cpp
)

target_link_libraries(gpu_visibility_checker
    ${catkin_LIBRARIES}
)

# Install rules
install(TARGETS gpu_visibility_checker
    ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
    DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
    FILES_MATCHING PATTERN "*.h"
)
