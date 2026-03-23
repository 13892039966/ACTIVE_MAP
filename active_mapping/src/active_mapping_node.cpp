#include <ros/ros.h>
#include "active_mapping/map_builder_base.h"
#include <memory>

int main(int argc, char** argv) {
    ros::init(argc, argv, "active_mapping_node");
    ros::NodeHandle nh("~");

    auto map_builder = std::make_unique<active_mapping::CloudMapBuilder>();
    map_builder->init(nh);
    map_builder->initParams(nh);

    map_builder->run();
    return 0;
}
