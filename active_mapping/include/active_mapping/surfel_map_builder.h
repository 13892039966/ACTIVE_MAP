#pragma once

#include "active_mapping/map_builder_base.h"

namespace active_mapping {

class SurfelMapBuilder : public MapBuilderBase {
public:
    SurfelMapBuilder() = default;
    ~SurfelMapBuilder() override = default;

    // 初始化范式 B 专属参数
    void initParams(ros::NodeHandle& nh);

protected:
    // 覆写基类纯虚函数：面元融合逻辑
    void processAndInsertCloud(CloudT::Ptr& cloud) override;
    
    // 面元地图的发布逻辑
    void publishMap() override;

private:
    float surfel_radius_;      // 面元覆盖半径
    float surfel_radius_sq_;   // 半径的平方 (加速计算)
    float depth_tolerance_;    // 点到面元的切平面距离容忍度 (越小越严格)
    int init_pca_k_;           // 仅在生成新面元时使用的 PCA 邻居数
    
    double w_geom_;            // 几何/边界杂乱度权重
    double w_hole_;            // 空洞/覆盖率缺失权重
};

} // namespace active_mapping