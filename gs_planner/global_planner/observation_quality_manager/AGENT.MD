# AGENT Brief · observation_quality_manager

本包负责维护世界内每个体素的观测质量、增量聚类自由空间、并为“需要好视角重新观测的区域”规划分层次的全局路径。下文聚焦于“我要改什么”和“该到哪里看”。

## 1. 核心使命
- 将同步到的有/无颜色点云体素化，增量更新几何复杂度、纹理复杂度与观测方向分布。
- 通过 GPU SYCL 双层 DDA 光线投射批量检查可见性，持续收紧“哪些方向仍可观测某体素”。
- 在 6 m Region 上聚类连通的自由空间，定位邻近观测质量差的障碍体素。
- 基于拓扑骨架 (TopoGraph) + LKH 求解器，先对候选 cluster 求粗粒度 TSP，再对第一个 cluster 内的视点做细粒度 TSP，输出可执行路径与朝向。
- 以 RViz/Marker 形式暴露几何、纹理、观测质量与自由空间状态，并提供 Python 直方图脚本快速质检。

## 2. 代码地图
| 目录 | 作用 |
| --- | --- |
| `include/observation_quality_manager/observation_quality_manager.h` | `VoxelCell`/`FreeRegion`/`ObservationQualityManager` 等核心数据结构与接口 |
| `src/observation_quality_manager.cpp` | 体素增量更新、自由空间 raycast、Region 聚类维护 |
| `include + src/global_planner.*` | 聚类筛选、GPU 评分、TSP 规划、可视化 |
| `include + src/gpu_raycast.*` | SYCL 两阶段 DDA、MxN CSR 可见性流水线 |
| `include + src/visibility_checker.*` | CPU↔GPU 数据打包、密集索引、暴露 `checkVisibilityGPU` |
| `include + src/oqm_visualization.*`, `visualization_utils.*` | RViz 点云/Marker 生成、自由空间可视化 |
| `src/oqm_node.cpp` | ROS 节点，订阅点云+里程计，定时刷新拓扑图、启动规划与可视化 |
| `scripts/plot_histograms.py` | 订阅 `/geometry_complexity_data` 等话题并实时画直方图 |
| `CLAUDE.md` | 另一份深入阅读顺序指南；本 AGENT 文档聚焦任务概览 |

## 3. 观测质量管理
- **体素建模 (`VoxelCell`)**
  - `geometry_occupancy_mask` / `texture_occupancy_mask`：10×10×10 子体素遮挡掩码，避免重复采样与做精细遮挡测试。
  - `SphericalBinning`（正二十面体 20 面）用于记录“观测过的方向”和“仍可用方向”，并借 LUT 评分。
  - `geometric_complexity` 通过 PCA surface-variation 归一化；`texture_complexity` 为 RGB 方差归一化；`observation_score` 只有在出现新方向时才增量刷新。
  - `available_direction_mask` 先默认全 0，GPU raycast 通过 `|=` 累积可见方向（不可见 → 可见），表示“在候选视点评估中，被 raycast 证实可见过的方向集合”（会跨轮次累计）。
  - `well_observed` 在 `VoxelCell::updateWellObserved()` 判断：`observation_score > 1.0 + 0.5*texture_complexity + 0.5*geometric_complexity` 或 `observation_score >= max_possible_score && max_possible_score > 0`。

- **点云融入 (`ObservationQualityManager::addPointCloudOnlyGeometry` / `addPointCloudWithTexture`)**
  - 使用体素索引哈希 (`spatial_hash_`) 分组，同一 voxel 批量处理，减少锁和重复 PCA。
  - 仅处理 `lidar_perception/max_ray_length` 范围内的点；几何更新结束后对每个点执行 `raycastFreeSpace`，将经过的 free voxel 写入 `region_map_` 并记录 `pending_modified_regions_`。
  - 彩色点云只更新纹理统计，不再重复 raycast。

- **Region 聚类 (`FreeRegion::computeClusterCenters`)**
- Region 边长 6 m (30 个体素)，使用 `voxel_states`（free/frontier/occupied，缺省为 unknown）。
  - 用 6 连通 BFS 只对“远离障碍的 safe voxel”聚类，并缓存与观测差障碍相邻的膨胀层，cluster 记录所有邻近的 poorly observed occupied voxel。
  - `ObservationQualityManager::updatePendingClusters()` 会在全局规划前触发，集中刷新受到 raycast 影响的 Region。

## 4. GPU 可见性流水线
- 文件：`src/gpu_raycast.cpp` / `include/.../gpu_raycast.h`，统一由 SYCL 编译（AdaptiveCpp `acpp` 优先、其次 oneAPI `dpcpp`；详见 `CMakeLists.txt`）。
- **两层 DDA**
  1. **粗体素 (0.2 m)**：沿光线执行 Amanatides-Woo DDA，只在遇到占据体素时进入第二层。
  2. **子体素 (0.02 m)**：对命中的粗体素内部执行细粒度 DDA，并借 `geometry_occupancy_mask` 位运算判断遮挡，允许穿过目标子体素本身。

- **MxN CSR 检查 (`gpu_raycast::launchMxNVisibilityCheckKernelAsync`)**
  - `VisibilityCSR` 同时给出 viewpoint→targets 以及 target→viewpoints 的稀疏结构，方便后续贪心覆盖与掩码更新。
  - Stage 流水线：密集索引建表 → Raycast + 原子计数 → prefix-sum 生成 offset → 写入 CSR → 更新 target available mask → 计算 `max_possible_score`。
  - 设备内存使用 `sycl::malloc_shared`，避免拷贝；评分表由 `SphericalBinning::scoring_table` 写入 400 float LUT。

## 5. 分层全局规划 (`global_planner.*`)
1. **Cluster 采集 (`collectAndFilterClusters`)**：遍历所有 Region 的 `FreeRegion::clusters_`，仅保留 `voxel_count > min_voxel_count` 且存在 poorly-observed 邻居的条目；按邻居数排序截断 `max_clusters`。
2. **粗粒度 TSP**
   - `buildDistanceMatrix` 使用 `TopoGraph::graphSearch` 计算 odom→cluster、cluster 互相 A* 距离，写入距离矩阵后交给 `lkh_tsp_solver`。
   - 不可达 cluster 使用 1000 的惩罚并在求解前过滤，保持 LKH 规模最小化。
3. **细粒度视点规划**（仅作用于 TSP 后的第一个 cluster）
   - `sampleViewpointsFromCluster`：对 Cluster free voxel 采样 (PCL `VoxelGrid`)，再调用 `LIOInterface::getDisToOcc` 剔除过近点。
   - `evaluateViewpointsWithGPU`：收集 target voxel + 区域内其余占据体素为潜在遮挡，调用 `checkVisibilityGPU` 得到 CSR 与更新后的可见性掩码。
   - `selectTopNViewpoints`：基于 yaw×pitch 网格（10×5）和相机 FOV 半径执行贪心最大覆盖，同时记录最佳 yaw/pitch（含 wrap-around 处理）。
4. **局部路径嵌套 TSP (`buildDistanceMatrixForViewpoints`)**
   - 若 odom Region 与视点 Region 26 邻接，则在局部膨胀后的 `LocalGrid` 内执行 grid A*；否则退回拓扑 A*。
   - 支持“视点→第二个 cluster”拼接，保证局部精细路径与其后的粗 TSP 无缝衔接。
5. **执行前更新掩码**：若 odom 已位于第一个精选视点附近（位置 <0.3 m、yaw/pitch <10°），调用 `updateVoxelMasksForViewpoint` 立即把实际观测写回 `observation_direction_mask`。
6. **可视化 (`planner_visualization.cpp`)**：`MarkerArray` 发布 TSP 路径、cluster 球、视点及箭头，同时可发布 viewpoint→target 的连线样式。

## 6. ROS 接入与数据流 (`src/oqm_node.cpp`)
1. **订阅**（近似时间同步）  
   - `/world_cloud_colored` (`sensor_msgs/PointCloud2`)  
   - `/world_cloud_nocolor` (`sensor_msgs/PointCloud2`)  
   - `/simOdometry` (`nav_msgs/Odometry`)
2. **回调流程**
   - 合并彩色/无色点云 → `ObservationQualityManager::addPointCloudOnlyGeometry`（几何+自由空间）。
   - 彩色点云单独调用 `addPointCloudWithTexture`，同步更新 `LIOInterface` 的地图姿态。
   - 更新 `latest_odom_*`，供规划与可视化使用。
3. **定时器**
   - 5 Hz：`OQMVisualization::publishAll` 输出点云/Marker。
   - 1 Hz：刷新 `TopoGraph`（`updateSkeleton` / `updateOdomNode` 等）→ `updatePendingClusters` → `GlobalPlanner::planGlobalTSPPath`。

### 发布话题速查
| 话题 | 类型 | 内容 |
| --- | --- | --- |
| `/geometry_complexity_visualization` | `sensor_msgs/PointCloud2` | 点颜色代表几何复杂度 |
| `/texture_complexity_visualization` | `sensor_msgs/PointCloud2` | 纹理复杂度 |
| `/observation_score_visualization` | `sensor_msgs/PointCloud2` | 当前观测质量分数 |
| `/well_observed_visualization` | `sensor_msgs/PointCloud2` | 是否满足 well_observed |
| `/free_space_visualization` | `sensor_msgs/PointCloud2` | 自由体素 |
| `/free_cluster_centers` | `visualization_msgs/Marker` | Cluster 球；颜色区分邻域观测质量 |
| `/global_planner/tsp_path` | `visualization_msgs/MarkerArray` | TSP 路径、视点等 |
| `/geometry_complexity_data` 等 | `std_msgs/Float32MultiArray` | Histogram 脚本输入 |

## 7. 构建与运行
1. **依赖**
   - ROS + catkin、PCL、OpenCV、Eigen、OpenMP。
   - `lidar_map`, `pointcloud_topo`, `lkh_tsp_solver` 自定义包。
   - SYCL 编译器：默认优先探测 `/usr/bin/acpp` (AdaptiveCpp)，否则 `dpcpp` (oneAPI)。确保可访问 CUDA runtime（`libcudart`），否则 CMake 会直接报错。
2. **编译**
   ```bash
   cd ~/catkin_ws
   catkin build observation_quality_manager
   ```
   CMake 会先用 SYCL 编译 `src/gpu_raycast.cpp` 生成 `gpu_raycast.o`，再与其余 C++17 目标链接。
3. **运行节点**
   ```bash
   source devel/setup.bash
   rosrun observation_quality_manager oqm_node
   ```
   如需直方图监控，可另启：
   ```bash
   rosrun observation_quality_manager plot_histograms.py
   ```

## 8. 开发/调试提示
- **确认体素边界**：`ObservationQualityManager::init` 会从 `LIOInterface` 读取 `global_map_min/max_boundary_`，若地图框架变化记得同步。
- **Region 聚类滞后**：只有调用 `updatePendingClusters()` 才会真正计算 BFS cluster；若在外部 API 要求实时结果需手动触发。
- **GPU 资源占用**：`checkVisibilityGPU` 会根据 target 数量在 shared memory 分配 `M×N` 缓冲，调试前可先在 `GlobalPlanner::evaluateViewpointsWithGPU` 对 `viewpoints`/`targets` 做限幅。
- **TSP 结果与路径对应**：`selected_views_` 会在 Fine TSP 后被重新排序；若要对特定视点做下一步处理，请使用 Fine TSP 返回的顺序。
- **全局路径刷新条件**：`planGlobalTSPPath` 失败会清空 `global_path_` 并发布空 Marker；可通过 `ROS_INFO` 输出查看 cluster/视点数、可达性原因。
- **可视化性能**：`VisualizationUtils` 会遍历整个 `spatial_hash_`；密度大时可调低定时器频率或在生成函数中加筛选。

## 9. 快速扩展入口
- 新的观测指标：可在 `VoxelCell` 中补充字段，并在 `addGeometryPoints` / `addTexturePoints` 中更新；注意同步到 GPU 版 `GPUVoxel`，否则 `checkVisibilityGPU` 无法读取。
- 多 cluster 精细化：当前仅对 TSP 第一个 cluster 细化，如果要对更多 cluster 做视点挑选，可在 `planGlobalTSPPath` 中循环调用 `sampleViewpointsFromCluster` + `buildDistanceMatrixForViewpoints` 并拼接结果。
- 替换 TSP 求解器：`tsp_calculation.cpp` 中 `solveTSP` 负责写 TSPLIB 文件并调用 `lkh_tsp_solver`，如需改用其他求解器可在此改写输出/解析逻辑。
- 引入新的可视化：`VisualizationUtils` 已封装 Jet colormap 和 Marker 生成，可在 `OQMVisualization::publishAll` 中按需添加。

> 小结：若你要修改“观测质量评价”“自由空间聚类”或“全局巡视任务”，分别进入 `observation_quality_manager.*`、`FreeRegion` 相关段落或 `global_planner.*`。GPU 逻辑集中在 `gpu_raycast.*` + `visibility_checker.*`，任何体素字段变动务必保持 CPU/GPU 结构一致。
