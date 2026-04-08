#include <gpu_visibility_checker/spherical_binning.h>

namespace gpu_visibility_checker {

void SphericalBinning::init() {
    if (!face_normals.empty()) return; // 已初始化

    // 1. 定义12个顶点 (0, ±1, ±PHI) 的循环排列
    Eigen::Vector3f verts[12] = {
      Eigen::Vector3f(-1,  PHI, 0), Eigen::Vector3f( 1,  PHI, 0),
      Eigen::Vector3f(-1, -PHI, 0), Eigen::Vector3f( 1, -PHI, 0),
      Eigen::Vector3f( 0, -1,  PHI), Eigen::Vector3f( 0,  1,  PHI),
      Eigen::Vector3f( 0, -1, -PHI), Eigen::Vector3f( 0,  1, -PHI),
      Eigen::Vector3f( PHI, 0, -1), Eigen::Vector3f( PHI, 0,  1),
      Eigen::Vector3f(-PHI, 0, -1), Eigen::Vector3f(-PHI, 0,  1)
    };

    // 2. 定义20个面的顶点索引 (标准拓扑结构)
    int indices[20][3] = {
      {0,11,5}, {0,5,1}, {0,1,7}, {0,7,10}, {0,10,11},
      {1,5,9}, {5,11,4}, {11,10,2}, {10,7,6}, {7,1,8},
      {3,9,4}, {3,4,2}, {3,2,6}, {3,6,8}, {3,8,9},
      {4,9,5}, {2,4,11}, {6,2,10}, {8,6,7}, {9,8,1}
    };

    // 3. 计算每个面的中心点并归一化 -> 得到20个方向向量
    face_normals.reserve(20);
    for (int i = 0; i < 20; ++i) {
      Eigen::Vector3f normal = Eigen::Vector3f::Zero();
      for (int k = 0; k < 3; ++k) {
        normal += verts[indices[i][k]];
      }
      // 归一化
      normal.normalize();
      face_normals.push_back(normal);
    }

    // 4. 预计算 20x20 评分表
    for (int i = 0; i < 20; ++i) {
      for (int j = 0; j < 20; ++j) {
        if (i == j) {
          // 法向量自己的方向评分固定为1
          scoring_table[i][j] = 1.0f;
        } else {
          // 其他方向评分为点积的绝对值
          scoring_table[i][j] = std::abs(face_normals[i].dot(face_normals[j]));
        }
      }
    }
}

int SphericalBinning::get_bin_index(const Eigen::Vector3f& view_dir) {
    // 数值稳定性处理：输入向量归一化
    float len_sq = view_dir.squaredNorm();
    if (len_sq < 1e-6f) return -1; // 无效观测（相机在Voxel内部）

    Eigen::Vector3f normalized = view_dir.normalized();

    int best_idx = 0;
    float max_dot = -2.0f;

    // 遍历20个方向求点积最大值
    for (int i = 0; i < 20; ++i) {
      float dot = normalized.dot(face_normals[i]);
      if (dot > max_dot) {
        max_dot = dot;
        best_idx = i;
      }
    }
    return best_idx;
}

const Eigen::Vector3f& SphericalBinning::get_face_normal(int bin_idx) {
    return face_normals[bin_idx];
}

} // namespace gpu_visibility_checker
