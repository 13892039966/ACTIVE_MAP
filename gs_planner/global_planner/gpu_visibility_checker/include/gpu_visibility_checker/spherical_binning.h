#pragma once
#include <Eigen/Eigen>
#include <vector>
#include <array>

namespace gpu_visibility_checker {

// ==========================================
// SphericalBinning - 正二十面体方向分箱
// ==========================================
class SphericalBinning {
private:
  // 黄金比例常数
  static constexpr float PHI = 1.618033988749895f; // (1 + sqrt(5)) / 2

  // 存储20个面的归一化中心向量
  inline static std::vector<Eigen::Vector3f> face_normals;

public:
  inline static std::array<std::array<float, 20>, 20> scoring_table;

  // 初始化：在程序启动时运行一次
  static void init();

  // 核心函数：输入观测向量，返回 0-19 的 Bin 索引
  static int get_bin_index(const Eigen::Vector3f& view_dir);

  // 获取指定bin的面法向量
  static const Eigen::Vector3f& get_face_normal(int bin_idx);
};

} // namespace gpu_visibility_checker
