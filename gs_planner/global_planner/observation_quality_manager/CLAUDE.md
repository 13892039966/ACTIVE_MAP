# Observation Quality Manager
使用 catkin build 编译

## 快速开始：代码阅读顺序

**按照以下顺序阅读代码，可以循序渐进地理解整个系统：**

### 第一步：理解基础数据结构（建议阅读时间：30分钟）
1. **gpu_types.h** (107行) - GPU数据结构定义
   - `Float3`: 对齐的3D向量
   - `GPUVoxel`: GPU端体素数据（位置、属性、状态、子体素掩码）
   - `GridInfo`: 3D密集索引网格信息
   - `VisibilityCSR`: 双向CSR稀疏可见性矩阵
   - **关键点**: 理解内存对齐（alignas(16)）和数据布局

2. **observation_quality_manager.h** 的数据结构部分 (行1-422)
   - `SphericalBinning`: 正二十面体方向分箱（20个均匀分布方向）
     - 关键函数: `get_bin_index()` - 将观测方向映射到0-19的bin索引
   - `VoxelHash`: 空间哈希函数
   - `VoxelCell`: CPU端体素数据结构
     - 关键字段: `normal`, `geometric_complexity`, `texture_complexity`, `observation_score`
     - 关键函数: `addGeometryPoints()`, `addTexturePoints()`, `updateObservationScore()`
   - `FreeRegion`: 自由空间区域管理（6m³）

### 第二步：理解核心算法（建议阅读时间：1小时）
3. **gpu_raycast.cpp** - 两层DDA可见性检查算法
   - **核心函数**: `RayCast(start, end, voxels, d_grid, grid_info)`
     - 第一层: 粗分辨率DDA (0.2m) - 遍历光线经过的体素
     - 第二层: 细分辨率DDA (0.02m) - 检查sub_masks遮挡
   - **关键函数**: `checkSubVoxelOcclusion()` - 子体素级遮挡检查
   - **MxN批量可见性检查**: `checkVisibilityGPU()` - 7个Stage流水线
     - Stage 1-2: 初始化和构建密集索引
     - Stage 3: Raycast并原子计数
     - Stage 4-5: Prefix-sum生成CSR格式
     - Stage 6-7: 计算available_mask和max_possible_score
   - **阅读建议**: 先看注释中的算法流程，再看实现细节



### 第三步：理解主要流程（建议阅读时间：45分钟）
5. **oqm_node.cpp** - ROS节点入口
   - **初始化流程**: `OQMNode::OQMNode()` - 创建OQM和可视化模块
   - **数据流**: `cloudCallback()` - 点云订阅回调
     1. 接收彩色点云 + 无色点云 + 里程计
     2. 合并点云 → `addPointCloudOnlyGeometry()` - 更新几何
     3. 彩色点云 → `addPointCloudWithTexture()` - 更新纹理
   - **可视化**: `visualizationTimerCallback()` - 5Hz定时发布
   - **阅读建议**: 从main()函数开始，追踪数据流

6. **observation_quality_manager.cpp** - 核心管理器实现
   - **批量几何更新**: `addPointCloudOnlyGeometry()`
     - 体素化点云 → 分批 → `VoxelCell::addGeometryPoints()`
     - 光线投射标记自由空间: `raycastFreeSpace()`
   - **批量纹理更新**: `addPointCloudWithTexture()`
     - 体素化彩色点云 → 分批 → `VoxelCell::addTexturePoints()`
   - **Region管理**: `updateClustersForRegions()` - 连通分量聚类
   - **阅读建议**: 重点关注批量处理和增量更新逻辑

### 第四步：理解高级功能（建议阅读时间：30分钟）
7. **global_planner.h + global_planner.cpp** - TSP全局路径规划
   - **主函数**: `planGlobalTSPPath()` - 完整的规划流程
     1. 收集并过滤clusters: `collectAndFilterClusters()`
     2. 层级规划（仅对第一个cluster）：
        - 从cluster的free voxels中降采样视点
        - 使用GPU raycast评分视点: `evaluateViewpointsWithGPU()`
        - 选择Top N视点: `selectTopNViewpoints()`
     3. 构建距离矩阵: `buildDistanceMatrix()` / `buildDistanceMatrixWithViewpoints()`
     4. 求解TSP: `solveTSP()` - 调用LKH求解器
   - **阅读建议**: 理解层级规划的思想（cluster级 + viewpoint级）

8. **oqm_visualization.cpp + visualization_utils.cpp** - 可视化工具
   - **主函数**: `OQMVisualization::publishAll()`
     - 发布体素可视化（几何/纹理复杂度、观测质量分数、well_observed状态）
     - 发布法向量箭头、自由空间clusters、全局路径等
   - **阅读建议**: 可选阅读，用于理解RViz可视化

### 调用流程总结图
```
oqm_node.cpp (ROS节点)
    ↓ cloudCallback()
ObservationQualityManager
    ↓ addPointCloudOnlyGeometry() / addPointCloudWithTexture()
VoxelCell (批量更新)
    ↓ addGeometryPoints() / addTexturePoints()
SphericalBinning (方向分箱)
    ↓ updateObservationScore()
gpu_raycast.cpp (GPU并行计算)
    ↓ checkVisibilityGPU()
    ↓ 更新 available_mask 和 max_possible_score
GlobalPlanner (路径规划，用户触发)
    ↓ planGlobalTSPPath()
    ↓ evaluateViewpointsWithGPU()
    ↓ solveTSP()
OQMVisualization (可视化，5Hz)
```

---

## 核心架构
**技术栈**：SYCL (跨平台GPU计算，支持AdaptiveCpp和Intel oneAPI)

### 系统组成
1. **VoxelCell**: 体素数据结构（CPU端）
2. **GPUVoxel**: GPU端体素数据结构
3. **GPU Raycast**: 两层DDA可见性检查（粗0.2m + 细0.02m）
4. **VisibilityChecker**: CPU-GPU交互层
5. **SphericalBinning**: 正二十面体方向分箱（20个均匀分布方向）

---

## 核心架构详解

### VoxelCell
**文件位置**: `include/observation_quality_manager/observation_quality_manager.h:139-382`

体素单元，存储几何、纹理、观测质量信息：

**几何属性**
- `normal`: 法向量 (PCA最小特征向量)
- `geometric_complexity`: 几何复杂度 (Surface Variation: λ_min/Σλ)
  - 原始范围 [0, 1/3]，归一化到 [0, 1]（乘以3）
  - 0 = 完美平面，1 = 各向同性分布
- 10x10x10子空间占用掩码，避免重复采样

**纹理属性**
- `texture_complexity`: RGB方差和
  - 原始范围 [0, 0.75]，归一化到 [0, 1]（乘以4/3）
  - 0 = 纯色，1 = 理论最大方差
- 增量式统计: `sum_color`, `sum_color_sq`
- 10x10x10子空间占用掩码

**观测质量**
- `observation_direction_mask`: 20方向观测掩码 (bitset<20>) - 已观测的方向
- `observation_score`: 当前观测质量分数
- `available_direction_mask`: 20方向可用观测掩码 (bitset<20>) - 可见的方向（GPU raycast计算）
  - 初始化为全0，表示“尚未确认任何方向可见”
  - **单向更新**：只能从不可见→可见（OR 操作，按方向累计被 raycast 证实可见过的 bin）
  - 用于计算 `max_possible_score`，并辅助判断“在候选视点评估的可见方向集合内是否已充分观测”
- `max_possible_score`: 在当前 `available_direction_mask` 上可达到的最大分数（由 GPU raycast 计算并回传）
  - 在 `gpu_raycast.cpp` 的 Stage 7 计算（frontier voxel 会跳过更新）
- `well_observed`: 是否充分观测
  - 条件1: `observation_score > 1.0 + 0.5*texture_complexity + 0.5*geometric_complexity`
  - 条件2: `observation_score >= max_possible_score && max_possible_score > 0`（在已知可见方向集合内“打满分”）

### SphericalBinning
**文件位置**: `include/observation_quality_manager/observation_quality_manager.h:17-96`

正二十面体方向分箱 (20个均匀分布方向)：
- `get_bin_index()`: 将向量映射到最近bin (点积最大)
- 预计算20x20评分表：`scoring_table[normal_bin][obs_bin]`
  - 对角线 (i==j): 固定为1.0（法向量自己的方向）
  - 非对角线: `abs(face_normals[i].dot(face_normals[j]))`

### GPUVoxel
**文件位置**: `include/observation_quality_manager/gpu_types.h:18-39`

GPU端体素数据结构（对齐到16字节）：

**位置与几何** (16 bytes)
- `center`: Float3 体素中心（x, y, z, w）

**属性** (16 bytes)
- `geo_complexity`: 几何复杂度
- `tex_complexity`: 纹理复杂度
- `normal_bin_idx`: 法向量对应的球面分箱索引
- `current_score`: 当前观测分数

**状态** (12 bytes)
- `obs_mask`: uint32 观测方向掩码（20位）
- `available_mask`: uint32 可用观测方向掩码（20位）
- `well_observed`: int32 是否充分观测

**子体素精细掩码** (128 bytes)
- `sub_masks[32]`: uint32[32] 存储1000位子体素占用信息
  - 每个子体素 0.02m (voxel_size/10)
  - 用于细分辨率DDA遮挡检查

**输出结果** (8 bytes)
- `max_possible_score`: 理论最大可能观测分数（GPU计算）
- `padding`: 填充

## 观测质量计算

### 触发机制
- 仅在检测到**新观测方向**时更新分数
- 检查逻辑: `!observation_direction_mask[obs_bin_idx]`

### 算法流程
1. 计算观测方向: `view_dir = camera_pos - voxel_center`
2. 分箱观测方向，更新 `observation_direction_mask`
3. 若是新观测方向（该 bin 首次置位），调用 `updateObservationScore()`（无参数）重算分数
4. 调用 `updateWellObserved()`：结合阈值与 `max_possible_score` 更新 `well_observed`

### 评分规则
- `observation_score` 直接对已置位的观测方向求和：
  - `observation_score = Σ scoring_table[normal_bin_idx][obs_bin]`
  - `scoring_table` 的对角线为 1，其余为 `abs(dot(face_normals[i], face_normals[j]))`
  - `normal_bin_idx` 来自几何 PCA 法向，且会在 `updateGeometry(view_dir)` 中根据当前观测方向翻转（让法向朝向观测方向）

### 性能优化
- 增量更新：只在“新观测方向”出现时重算 `observation_score`（固定 20 个 bin，开销 O(20)）

## 数据过滤
- 距离过滤: 超过 `max_ray_length` 的点跳过
- 子空间去重: 每个子体素只保留一个点

## 增量式计算
- **几何**: 累积 `sum_pos_local`, `sum_pp_local` → 协方差矩阵 → 特征值分解
  - 归一化: `3.0 * λ_min / Σλ`（最大值1/3 → 1）
- **纹理**: 累积 `sum_color`, `sum_color_sq` → 方差公式: `Var = E[X²] - (E[X])²`
  - 归一化: `(4/3) * Σ(Var_RGB)`（最大值0.75 → 1）
- **局部坐标**: 使用 `p_local = p_global - voxel_center` 避免浮点精度丢失

## 归一化范围
| 指标 | 原始范围 | 归一化公式 | 最终范围 |
|------|----------|------------|----------|
| `geometric_complexity` | [0, 1/3] | `×3` | [0, 1] |
| `texture_complexity` | [0, 0.75] | `×4/3` | [0, 1] |

---

## GPU Raycast - 两层DDA可见性检查
**文件位置**: `src/gpu_raycast.cpp` + `include/observation_quality_manager/gpu_raycast.h`

### 核心思想
**两层DDA算法**：
1. **粗分辨率** (0.2m)：遍历光线经过的体素
2. **细分辨率** (0.02m)：在非空体素内检查sub_masks遮挡

### 算法流程
`RayCast(start, end, voxels, d_grid, grid_info)` → 返回 1=可见, 0=遮挡

**预处理**：
- 距离过滤：`dist_sq < 0.01` → 可见（自己看自己）
- 距离过滤：`dist_sq > 100` → 不可见（>10m）
- 计算目标点的粗体素坐标 `(target_gx, target_gy, target_gz)`
- 计算目标点在粗体素内的子体素坐标 `(target_sx, target_sy, target_sz)` [0-9]

**第一层：粗分辨率DDA** (0.2m)
1. 初始化：起点体素 `(x, y, z)`，终点体素 `(endX, endY, endZ)`
2. 计算步进方向 `(stepX, stepY, stepZ)` 和DDA参数 `(tMaxX/Y/Z, tDeltaX/Y/Z)`
3. 遍历光线经过的每个体素：
   - 使用 `getVoxelIndex()` 进行 **O(1)密集索引查询**
   - 如果体素非空 → 调用 `checkSubVoxelOcclusion()` 进行细分辨率检查
   - 如果被遮挡 → 返回0
   - DDA步进到下一个体素
4. 到达终点 → 返回1（未被遮挡）

**第二层：细分辨率DDA** (0.02m)
`checkSubVoxelOcclusion(ray_start, ray_end, v, voxel_size, target_sx/y/z)`

1. **坐标转换**：世界坐标 → 子体素坐标系 [0, 10)
2. **AABB裁剪**：将光线裁剪到粗体素边界内 `[0, 10)`
3. **细分辨率DDA遍历**：
   - 起点子体素 `(sx, sy, sz)` [0-9]
   - 遍历光线经过的每个子体素（最多30个）
   - 检查当前子体素是否为目标 → 是则跳过（不遮挡自己）
   - 检查 `sub_masks` 中对应位是否为1 → 是则返回true（被遮挡）
   - DDA步进到下一个子体素
4. 到达终点 → 返回false（未被遮挡）

**关键优化**：
- **O(1)密集索引**：通过3D数组 `d_grid[lx][ly][lz]` 直接查询体素索引
- **GridInfo结构**：存储网格边界和尺寸，避免越界访问
- **sub_masks位运算**：
  ```cpp
  int bit_idx = sx * 100 + sy * 10 + sz;  // 线性索引
  int array_idx = bit_idx / 32;           // uint32数组索引
  int bit_offset = bit_idx % 32;          // 位偏移
  bool occupied = (sub_masks[array_idx] >> bit_offset) & 1;
  ```

### MxN批量可见性检查（CSR输出）

**功能**：并行检查M个视点到N个目标的可见性，输出双向CSR稀疏矩阵

**输入**：
- `d_viewpoints[M]`: M个视点位置
- `d_target_indices[N]`: N个目标体素索引（指向d_voxels数组）
- `d_voxels[num_voxels]`: 所有体素数据
- `d_scoring_table[400]`: 20x20评分表（flatten）

**输出**（VisibilityCSR）：
1. **方向1: viewpoint → targets**
   - `viewpoint_to_targets[]`: 所有可见target索引（拼接）
   - `viewpoint_offsets[M+1]`: 每个viewpoint的起始位置
2. **方向2: target → viewpoints**
   - `target_to_viewpoints[]`: 所有能看到的viewpoint索引（拼接）
   - `target_offsets[N+1]`: 每个target的起始位置
3. **Target结果**：
   - `updated_available_masks[N]`: 每个target的available_mask
   - `updated_max_scores[N]`: 每个target的max_possible_score

**7个Stage流水线**：

**Stage 1: 初始化** (GPU)
- 并行初始化 `d_grid[grid_total_size] = -1`
- 并行初始化 `d_viewpoint_counts[M] = 0`
- 并行初始化 `d_target_counts[N] = 0`

**Stage 2: 构建voxel密集索引** (GPU)
```cpp
for (int i : num_voxels) {
    int gx = floor((voxel.center.x - map_origin_x) / voxel_size);
    int lx = gx - grid_info.origin_x;  // 转换为grid局部坐标
    int grid_idx = lx * size_y * size_z + ly * size_z + lz;
    d_grid[grid_idx] = i;  // 存储voxel索引
}
```

**Stage 3: Raycast并原子计数** (GPU)
```cpp
for (int i : M*N) {  // 并行MxN对
    int viewpoint_idx = i / N;
    int target_idx = i % N;
    int visibility = RayCast(viewpoint, target, ...);  // 两层DDA
    d_visibility_temp[i] = visibility;
    if (visibility) {
        atomic_add(d_viewpoint_counts[viewpoint_idx], 1);
        atomic_add(d_target_counts[target_idx], 1);
    }
}
```

**Stage 4: Prefix-sum生成offsets** (CPU)
```cpp
// viewpoint offsets
d_viewpoint_offsets[0] = 0;
for (int i : M) {
    d_viewpoint_offsets[i+1] = d_viewpoint_offsets[i] + d_viewpoint_counts[i];
}
// target offsets 同理
// 重置counts为0（用作写入位置计数器）
```

**Stage 5: 并行写入CSR数组** (GPU)
```cpp
for (int i : M*N) {
    if (d_visibility_temp[i]) {
        int pos1 = atomic_add(d_viewpoint_counts[viewpoint_idx], 1);
        d_viewpoint_to_targets[d_viewpoint_offsets[viewpoint_idx] + pos1] = target_idx;

        int pos2 = atomic_add(d_target_counts[target_idx], 1);
        d_target_to_viewpoints[d_target_offsets[target_idx] + pos2] = viewpoint_idx;
    }
}
```

**Stage 6: 计算available_mask（单向更新）** (GPU)
```cpp
for (int target_idx : N) {
    uint32_t current_visible_mask = 0;
    // 遍历能看到这个target的所有viewpoints
    for (int i : [d_target_offsets[target_idx], d_target_offsets[target_idx+1])) {
        int viewpoint_idx = d_target_to_viewpoints[i];
        Float3 view_dir = viewpoint - voxel_center;
        int bin_idx = getBinIndex(view_dir);
        current_visible_mask |= (1u << bin_idx);
    }
    // 单向更新：OR操作（不可见→可见，不可逆）
    d_voxels[voxel_idx].available_mask |= current_visible_mask;
}
```

**Stage 7: 计算max_possible_score** (GPU)
```cpp
for (int target_idx : N) {
    int normal_bin_idx = d_voxels[voxel_idx].normal_bin_idx;
    float score = 0.0f;
    uint32_t available_mask = d_voxels[voxel_idx].available_mask;
    for (int i : 20) {
        if (available_mask & (1u << i)) {
            score += d_scoring_table[normal_bin_idx * 20 + i];
        }
    }
    d_voxels[voxel_idx].max_possible_score = score;
}
```


## 数据结构补充

### GridInfo - 3D密集索引网格信息
**文件位置**: `include/observation_quality_manager/gpu_types.h:60-65`

```cpp
struct GridInfo {
    int origin_x, origin_y, origin_z;  // grid原点（格子索引，相对于map_origin）
    int size_x, size_y, size_z;        // grid尺寸（格子数）
    float voxel_size;                  // 体素大小（0.2m）
    float map_origin_x, map_origin_y, map_origin_z;  // 地图原点（世界坐标）
};
```
**用途**：将稀疏的体素哈希表转换为密集的3D数组，实现O(1)查询

### VisibilityCSR - 双向CSR稀疏可见性矩阵
**文件位置**: `include/observation_quality_manager/gpu_types.h:73-107`
```cpp
struct VisibilityCSR {
    // 方向1: viewpoint → targets
    vector<int> viewpoint_to_targets;     // 所有可见target索引（拼接）
    vector<int> viewpoint_offsets;        // [M+1] 每个viewpoint的起始位置

    // 方向2: target → viewpoints
    vector<int> target_to_viewpoints;     // 所有能看到的viewpoint索引（拼接）
    vector<int> target_offsets;           // [N+1] 每个target的起始位置

    // GPU计算结果
    vector<uint32_t> updated_available_masks;  // [N] 每个target的available_mask
    vector<float> updated_max_scores;          // [N] 每个target的max_possible_score

    // 查询接口
    vector<int> getVisibleTargets(int viewpoint_idx);     // O(可见数)
    vector<int> getObservingViewpoints(int target_idx);   // O(观测者数)
};
```

---

## 快速参考索引

### 核心类和结构体定位表
| 类/结构体 | 文件路径 | 行号范围 | 主要功能 |
|----------|---------|---------|---------|
| `GPUVoxel` | `gpu_types.h` | 18-39 | GPU端体素数据结构 |
| `VisibilityCSR` | `gpu_types.h` | 73-107 | 双向CSR稀疏可见性矩阵 |
| `GridInfo` | `gpu_types.h` | 60-65 | 3D密集索引网格信息 |
| `SphericalBinning` | `observation_quality_manager.h` | 17-96 | 20方向分箱系统 |
| `VoxelCell` | `observation_quality_manager.h` | 139-382 | CPU端体素数据结构 |
| `FreeRegion` | `observation_quality_manager.h` | 387-421 | 自由空间区域管理 |
| `ObservationQualityManager` | `observation_quality_manager.h` | 424-501 | 核心管理器类定义 |
| `VisibilityChecker` | `visibility_checker.h` | 38-76 | CPU-GPU交互层 |
| `GlobalPlanner` | `global_planner.h` | 28-143 | TSP全局路径规划 |
| `OQMVisualization` | `oqm_visualization.h` | - | RViz可视化模块 |

### 关键函数定位表
| 函数名 | 文件路径 | 主要功能 | 调用频率 |
|-------|---------|---------|---------|
| `main()` | `oqm_node.cpp:143` | 程序入口 | 启动时 |
| `cloudCallback()` | `oqm_node.cpp:36` | 点云数据回调 | 每帧 |
| `addPointCloudOnlyGeometry()` | `observation_quality_manager.cpp` | 批量几何更新 | 每帧 |
| `addPointCloudWithTexture()` | `observation_quality_manager.cpp` | 批量纹理更新 | 每帧 |
| `VoxelCell::addGeometryPoints()` | `observation_quality_manager.h:190` | 添加几何点 | 每体素每帧 |
| `VoxelCell::addTexturePoints()` | `observation_quality_manager.h:271` | 添加纹理点 | 每体素每帧 |
| `VoxelCell::updateObservationScore()` | `observation_quality_manager.h:344` | 更新观测分数 | 新观测时 |
| `SphericalBinning::get_bin_index()` | `observation_quality_manager.h:76` | 方向映射到bin | 频繁调用 |
| `checkVisibilityGPU()` | `gpu_raycast.cpp` | GPU批量可见性检查 | 可见性检查时 |
| `RayCast()` | `gpu_raycast.cpp` | 两层DDA光线投射 | GPU并行调用 |
| `GlobalPlanner::planGlobalTSPPath()` | `global_planner.cpp` | TSP路径规划 | 用户触发 |
| `OQMVisualization::publishAll()` | `oqm_visualization.cpp` | 发布所有可视化 | 5Hz定时 |

### 关键概念速查
| 概念 | 说明 | 相关代码位置 |
|-----|------|------------|
| **体素化** | 将点云映射到0.2m网格 | `observation_quality_manager.cpp`, `pos2idx()` |
| **子体素掩码** | 10×10×10=1000位掩码，避免重复采样 | `VoxelCell::geometry_occupancy_mask` |
| **20方向分箱** | 正二十面体均匀划分球面 | `SphericalBinning::face_normals` |
| **观测质量分数** | 基于观测方向与法向量关系评分 | `VoxelCell::observation_score` |
| **available_mask** | 可见方向掩码（GPU raycast计算） | `VoxelCell::available_direction_mask` |
| **well_observed** | 是否充分观测标志 | `VoxelCell::well_observed` |
| **两层DDA** | 粗0.2m + 细0.02m光线投射 | `gpu_raycast.cpp:RayCast()` |
| **CSR格式** | 压缩稀疏行，双向可见性查询 | `VisibilityCSR` |
| **自由空间聚类** | 6m³ region内连通分量聚类 | `FreeRegion::computeConnectedComponents()` |
| **层级规划** | Cluster级 + Viewpoint级TSP | `GlobalPlanner::planGlobalTSPPath()` |

### 数据流追踪要点
**点云 → 体素更新**:
```
ROS消息 (cloudCallback)
  ↓
pcl::PointCloud
  ↓ 体素化 (pos2idx)
std::unordered_map<体素索引, VoxelCell>
  ↓ 批量更新
VoxelCell::addGeometryPoints() / addTexturePoints()
  ↓ 子空间去重
geometry_occupancy_mask / texture_occupancy_mask
  ↓ 增量统计
sum_pos_local, sum_pp_local, sum_color, sum_color_sq
  ↓ 计算复杂度
geometric_complexity, texture_complexity
  ↓ 更新观测分数
observation_score
```

**可见性检查流程**:
```

std::vector<Eigen::Vector3i> nearby_voxel_indices
  ↓ 数据转换
VoxelCell → GPUVoxel (convertToGPUVoxel)
  ↓ GPU并行计算
checkVisibilityGPU() [7-Stage流水线]
  ↓ 返回CSR格式
VisibilityCSR {viewpoint_to_targets, target_to_viewpoints}
  ↓ 同步回CPU
更新 available_mask 和 max_possible_score
```

**TSP规划流程**:
```
GlobalPlanner::planGlobalTSPPath()
  ↓ 收集clusters
collectAndFilterClusters() [过滤 + 排序]
  ↓ 层级规划（仅cluster1）
sampleViewpointsFromCluster() → 降采样视点
evaluateViewpointsWithGPU() → GPU raycast评分
selectTopNViewpoints() → 选Top N
  ↓ 构建距离矩阵
buildDistanceMatrixWithViewpoints()
  ↓ 求解TSP
solveTSP() [调用LKH]
  ↓ 生成路径
global_path_ (odom → viewpoints → clusters)
```

### 性能关键点
1. **子空间去重**: 每个0.02m子体素只保留一个点，避免重复计算 (VoxelCell:197, 293)
2. **增量更新**: 累积统计量，避免每次重算 (VoxelCell:206-208, 302-304)
3. **新观测检测**: 只在新方向时更新分数 (VoxelCell:281, 309)
4. **O(1)密集索引**: 3D数组替代哈希查询 (gpu_raycast.cpp:Stage 2)
5. **CSR稀疏矩阵**: 避免存储M×N完整矩阵 (VisibilityCSR)
6. **GPU并行**: 单kernel处理M×N对可见性检查 (gpu_raycast.cpp:Stage 3)

### 调试技巧
1. **查看体素数据**: RViz订阅 `/oqm_voxels_*` 话题查看复杂度和观测质量
2. **检查可见性**: 订阅 `/oqm_visibility_lines` 查看光线投射结果
3. **验证法向量**: 订阅 `/oqm_normals` 查看法向量箭头
4. **追踪路径**: 订阅 `/oqm_global_path` 查看TSP规划结果
5. **性能分析**: 查看 `ROS_INFO` 输出的几何/纹理更新耗时
