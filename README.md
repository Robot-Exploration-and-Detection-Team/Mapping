# 建图与状态估计模块

基于四足机器人的危险源自主搜索与识别比赛 —— 建图与状态估计（mapping + state estimation）基础设施层。

负责把 LiDAR 点云 + IMU 转化为两项关键输出：

1. 机器人当前在哪 —— 状态估计 / 里程计
2. 周围环境长什么样 —— 栅格地图

## 目录结构

```
mapping_state_estimation/   我的完整 ROS 包（直接放进 catkin 工作区 src/ 即可）
FAST_LIO/                   第三方 FAST_LIO 的适配改动（8 个文件，需覆盖到上游仓库）
```

## 环境

- ROS1 Noetic + Gazebo Classic + Unitree A1
- 依赖：`sudo apt install ros-noetic-robot-localization`

## 部署步骤

### 1. mapping_state_estimation

把 `mapping_state_estimation/` 整个目录放进你的 catkin 工作区 `src/` 下，然后编译：

```bash
catkin_make -j
```

### 2. FAST_LIO（重要）

`FAST_LIO/` 里只有我改动的 8 个文件，**不是完整仓库**。先 clone 上游，再用本目录覆盖：

```bash
cd <catkin_ws>/src
git clone https://github.com/hku-mars/FAST_LIO.git
# 用本仓库的 FAST_LIO/ 覆盖 clone 出来的 FAST_LIO/
```

覆盖后编译：

```bash
catkin_make --build build_new --pkg fast_lio mapping_state_estimation
```

## 启动

```bash
# 基础模式（默认启用 FAST-LIO）
roslaunch mapping_state_estimation mapping_master.launch

# 含回环检测
roslaunch mapping_state_estimation mapping_master.launch enable_loop_closure:=true

# 低 CPU 模式
roslaunch mapping_state_estimation mapping_master.launch low_cpu_mode:=true
```

## 对外接口（队友订阅）

| 输出 | 类型 | 用途 |
|------|------|------|
| `/map` | nav_msgs/OccupancyGrid | 栅格地图 |
| `/odometry/filtered` | nav_msgs/Odometry | 融合里程计 |
| `/robot_pose` | geometry_msgs/PoseStamped | 机器人位姿 |
| `/current_floor` | std_msgs/Int8 | 当前楼层 |
| `world → odom` TF | tf2 | 全局修正 |
| `odom → base` TF | tf2 | 里程计位姿 |

## 注意

- 比赛禁止使用 `/ground_truth/*`、`/Odometry_gazebo`、`layout_metadata.json`、`danger_truth.json`。
