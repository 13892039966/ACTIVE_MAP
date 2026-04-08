cmake_minimum_required(VERSION 3.0.2)
project(observation_quality_manager)
find_package(OpenMP REQUIRED)

set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_BUILD_TYPE "RelWithDebInfo")
# set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-Wno-enum-constexpr-conversion ${OpenMP_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall")

# ==========================================
# 自动检测 SYCL 编译器 (acpp 或 dpcpp)
# ==========================================
find_program(ACPP_COMPILER acpp)
# Also search common oneAPI install locations in case PATH is not initialized.
find_program(DPCPP_COMPILER dpcpp
    HINTS
        /opt/intel/oneapi/compiler/latest/bin
        /opt/intel/oneapi/compiler/2025.3/bin
)

if(ACPP_COMPILER)
    set(SYCL_COMPILER ${ACPP_COMPILER})
    set(SYCL_FLAGS_STR "-std=c++17;-O2;-fPIC;--acpp-targets=cuda:sm_89;-fopenmp")
    message(STATUS "Found AdaptiveCpp: ${ACPP_COMPILER}")
    set(SYCL_INCLUDE_DIR "/opt/AdaptiveCpp/include/AdaptiveCpp")

    # 查找 CUDA 库 (AdaptiveCpp 需要)
    find_package(CUDA QUIET)
    if(CUDA_FOUND)
        set(SYCL_LIBRARIES ${CUDA_LIBRARIES} ${CUDA_cudart_static_LIBRARY})
        message(STATUS "Found CUDA: ${CUDA_VERSION}")
    else()
        # 尝试手动查找 libcudart
        find_library(CUDART_LIB cudart PATHS /usr/local/cuda/lib64 /opt/cuda/lib64)
        if(CUDART_LIB)
            set(SYCL_LIBRARIES ${CUDART_LIB})
            message(STATUS "Found CUDA runtime manually: ${CUDART_LIB}")
        else()
            message(WARNING "CUDA not found - linking may fail")
        endif()
    endif()

    # 查找 HipSYCL/AdaptiveCpp 运行时库 (必需！)
    find_library(ACPP_RT_LIB acpp-rt PATHS /opt/AdaptiveCpp/lib /usr/local/lib /usr/lib)
    if(ACPP_RT_LIB)
        list(APPEND SYCL_LIBRARIES ${ACPP_RT_LIB})
        message(STATUS "Found AdaptiveCpp runtime: ${ACPP_RT_LIB}")
    else()
        message(FATAL_ERROR "AdaptiveCpp runtime library (libacpp-rt.so) not found!")
    endif()

    # 添加 OpenMP 库（gpu_raycast.o 用 -fopenmp 编译，需要 LLVM OpenMP 运行时）
    # 直接链接 libomp.so.5（LLVM OpenMP）
    list(APPEND SYCL_LIBRARIES "/usr/lib/x86_64-linux-gnu/libomp.so.5")
    message(STATUS "Added LLVM OpenMP library: /usr/lib/x86_64-linux-gnu/libomp.so.5")

elseif(DPCPP_COMPILER)
    set(SYCL_COMPILER ${DPCPP_COMPILER})
    set(SYCL_FLAGS_STR "-std=c++17;-O2;-fPIC;-fsycl")
    # dpcpp must drive the final link step to keep SYCL device images/kernel metadata.
    set(USE_SYCL_LINK_DRIVER TRUE)
    message(STATUS "Found Intel oneAPI DPC++: ${DPCPP_COMPILER}")

    # When the final target is linked by g++, add oneAPI SYCL headers/libs explicitly.
    get_filename_component(DPCPP_REAL ${DPCPP_COMPILER} REALPATH)
    get_filename_component(DPCPP_BIN_DIR ${DPCPP_REAL} DIRECTORY)
    get_filename_component(DPCPP_BASE_DIR "${DPCPP_BIN_DIR}/../.." ABSOLUTE)

    find_path(SYCL_INCLUDE_DIR sycl/sycl.hpp
        PATHS
            "${DPCPP_BASE_DIR}/include"
            "/opt/intel/oneapi/compiler/latest/include"
        NO_DEFAULT_PATH
    )
    if(NOT SYCL_INCLUDE_DIR)
        message(FATAL_ERROR "SYCL headers not found for oneAPI (missing sycl/sycl.hpp).")
    endif()

    set(DPCPP_LIB_HINTS
        "${DPCPP_BASE_DIR}/lib"
        "/opt/intel/oneapi/compiler/latest/lib"
    )

    find_library(DPCPP_SYCL_LIB sycl PATHS ${DPCPP_LIB_HINTS} NO_DEFAULT_PATH)
    find_library(DPCPP_INTLC_LIB intlc PATHS ${DPCPP_LIB_HINTS} NO_DEFAULT_PATH)
    find_library(DPCPP_IRC_LIB irc PATHS ${DPCPP_LIB_HINTS} NO_DEFAULT_PATH)
    find_library(DPCPP_IMF_LIB imf PATHS ${DPCPP_LIB_HINTS} NO_DEFAULT_PATH)
    find_library(DPCPP_SVML_LIB svml PATHS ${DPCPP_LIB_HINTS} NO_DEFAULT_PATH)
    find_library(DPCPP_IRNG_LIB irng PATHS ${DPCPP_LIB_HINTS} NO_DEFAULT_PATH)

    if(NOT DPCPP_SYCL_LIB)
        message(FATAL_ERROR "libsycl not found for oneAPI.")
    endif()
    if(NOT DPCPP_INTLC_LIB)
        message(FATAL_ERROR "libintlc not found for oneAPI.")
    endif()

    # Keep order close to oneAPI linker driver to satisfy symbol resolution.
    set(SYCL_LIBRARIES
        ${DPCPP_SVML_LIB}
        ${DPCPP_IRNG_LIB}
        ${DPCPP_IMF_LIB}
        ${DPCPP_IRC_LIB}
        ${DPCPP_SYCL_LIB}
        -Wl,--no-as-needed
        ${DPCPP_INTLC_LIB}
        -Wl,--as-needed
    )
else()
    message(FATAL_ERROR "No SYCL compiler found! Please install AdaptiveCpp or Intel oneAPI.")
endif()

find_package(OpenCV REQUIRED)
find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    nav_msgs
    cv_bridge
    message_filters
    visualization_msgs
    lidar_map
    pointcloud_topo
    lkh_tsp_solver
    gpu_visibility_checker
)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED COMPONENTS segmentation)
catkin_package(
    INCLUDE_DIRS include
    LIBRARIES observation_quality_manager
    CATKIN_DEPENDS roscpp std_msgs nav_msgs lidar_map pointcloud_topo lkh_tsp_solver gpu_visibility_checker

)
include_directories(
    SYSTEM
    include
    ${catkin_INCLUDE_DIRS}
    ${Eigen3_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${SYCL_INCLUDE_DIR}
)
link_directories(${PCL_LIBRARY_DIRS})

# ==========================================
# GPU 代码编译 (gpu_raycast.cpp 用 SYCL 编译器单独编译)
# ==========================================
# 构建 include 路径列表
set(GPU_INCLUDE_FLAGS)
list(APPEND GPU_INCLUDE_FLAGS "-I${CMAKE_CURRENT_SOURCE_DIR}/include")
foreach(dir ${catkin_INCLUDE_DIRS})
    list(APPEND GPU_INCLUDE_FLAGS "-I${dir}")
endforeach()
foreach(dir ${OpenMP_CXX_INCLUDE_DIRS})
    list(APPEND GPU_INCLUDE_FLAGS "-I${dir}")
endforeach()

# Only add hard-coded fallback OpenMP header paths for AdaptiveCpp.
# With oneAPI dpcpp these GCC/LLVM include directories can override the
# compiler's builtin headers and break SIMD intrinsic resolution.
if(ACPP_COMPILER)
    list(APPEND GPU_INCLUDE_FLAGS "-I/usr/lib/gcc/x86_64-linux-gnu/10/include")
    list(APPEND GPU_INCLUDE_FLAGS "-I/usr/lib/llvm-10/include/openmp")
endif()

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/gpu_raycast.o
    COMMAND ${SYCL_COMPILER} ${SYCL_FLAGS_STR} ${GPU_INCLUDE_FLAGS}
            -c ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu_raycast.cpp
            -o ${CMAKE_CURRENT_BINARY_DIR}/gpu_raycast.o
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu_raycast.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/include/observation_quality_manager/gpu_raycast.h
            ${CMAKE_CURRENT_SOURCE_DIR}/include/observation_quality_manager/gpu_types.h
    COMMENT "Compiling gpu_raycast.cpp with SYCL compiler (${SYCL_COMPILER})"
)

# 创建一个目标来触发 GPU 代码编译
add_custom_target(gpu_raycast_obj ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/gpu_raycast.o)

# When C++ sources are compiled by g++, force dpcpp to link this package's targets so
# gpu_raycast.o keeps its SYCL offload bundle in the final shared library.
if(USE_SYCL_LINK_DRIVER)
    set(CMAKE_CXX_CREATE_SHARED_LIBRARY
        "${SYCL_COMPILER} -fsycl <CMAKE_SHARED_LIBRARY_CXX_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS> <SONAME_FLAG><TARGET_SONAME> -o <TARGET> <OBJECTS> <LINK_LIBRARIES>")
    set(CMAKE_CXX_LINK_EXECUTABLE
        "${SYCL_COMPILER} -fsycl <FLAGS> <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")
endif()

# ==========================================
# 主库 (其他所有文件, 用 g++ 编译, 链接 GPU object file)
# ==========================================
add_library(observation_quality_manager
    src/observation_quality_manager.cpp
    src/visualization_utils.cpp
    src/visibility_checker.cpp
    src/oqm_visualization.cpp
    src/global_planner.cpp
    src/viewpoint_extracter.cpp
    src/region_dilation.cpp
    src/watershed_subdivision.cpp
    src/tsp_calculation.cpp
    src/planner_visualization.cpp
    # SOP Solver files
    src/sop_solver/sop_solver_interface.cpp
    src/sop_solver/solver.cpp
    src/sop_solver/digraph.cpp
    src/sop_solver/hungarian.cpp
    src/sop_solver/edge.cpp
    src/sop_solver/solver_state.cpp
    src/sop_solver/history_node.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/gpu_raycast.o  # 链接 GPU object file
)

# 确保 GPU 代码先编译
add_dependencies(observation_quality_manager gpu_raycast_obj)

target_link_libraries(observation_quality_manager PUBLIC
    ${catkin_LIBRARIES}
    ${PCL_LIBRARIES}
    ${OpenCV_LIBS}
    OpenMP::OpenMP_CXX
    ${SYCL_LIBRARIES}  # CUDA runtime for GPU code
    stdc++
)

# Install Python scripts
catkin_install_python(PROGRAMS
    scripts/plot_histograms.py
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
