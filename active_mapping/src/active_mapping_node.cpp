#include <ros/ros.h>
#include "active_mapping/cloud_map_builder.h"
#include "active_mapping/surfel_map_builder.h"
#include <memory>

int main(int argc, char** argv) {
    ros::init(argc, argv, "active_mapping_node");
    ros::NodeHandle nh("~");

    std::string map_type;
    nh.param<std::string>("map_type", map_type, "pointcloud");

    // 核心架构：多态调度
    std::unique_ptr<active_mapping::MapBuilderBase> map_builder;

    if (map_type == "surfel") {
        auto surfel_builder = std::make_unique<active_mapping::SurfelMapBuilder>();
        surfel_builder->init(nh);       // 基类公共初始化
        surfel_builder->initParams(nh); // 派生类 B 初始化
        map_builder = std::move(surfel_builder);
    } else if (map_type == "pointcloud"){
        auto cloud_builder = std::make_unique<active_mapping::CloudMapBuilder>();
        cloud_builder->init(nh);       // 基类公共初始化
        cloud_builder->initParams(nh); // 派生类 A 初始化
        map_builder = std::move(cloud_builder);
    }

    if (!map_builder) {
        ROS_ERROR("Unsupported map_type: %s", map_type.c_str());
        return 1;
    }

    map_builder->run();
    return 0;
}
