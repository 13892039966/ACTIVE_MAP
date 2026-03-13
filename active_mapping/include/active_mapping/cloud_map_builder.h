#pragma once

#include "active_mapping/map_builder_base.h"

namespace active_mapping {

class CloudMapBuilder : public MapBuilderBase {
public:
    CloudMapBuilder() = default;
    ~CloudMapBuilder() override = default;

    void initParams(ros::NodeHandle& nh);

protected:
    void processAndInsertCloud(CloudT::Ptr& cloud) override;
    void publishMap() override;

private:
    // 🌟 新增：独立子地图发布器
    ros::Publisher pub_geom_map_;   // 几何熵热力图
    ros::Publisher pub_color_map_;  // 色彩熵热力图

    int knn_k_;
    double w_geom_;
    double w_color_;
    double w_sparse_;
    bool compute_normals_;
    bool publish_geom_map_;
    bool publish_color_map_;
    CloudT cached_cloud_;
    bool cache_valid_ = false;
};

} // namespace active_mapping
