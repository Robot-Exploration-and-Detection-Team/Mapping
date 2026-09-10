#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
 2D 占据栅格地图生成节点 (Step 7 + Step 8)
==============================================================================
 功能: 将 3D LiDAR 点云投影为 2D 占据栅格地图，供路径规划使用。

 处理流程:
   1. 点云高度过滤: 取 z ∈ [0.1, 1.5] 范围内的点
   2. 投影到 2D: 忽略 z 坐标，投影到 XY 平面
   3. 基于机器人位姿将点云变换到 world 坐标系
   4. log-odds 占据概率更新 (hit/miss)
   5. 多楼层管理 (Step 8): 根据机器人 z 坐标切换活跃楼层
   6. 发布 nav_msgs/OccupancyGrid

 订阅:
   /cloud_base              sensor_msgs/PointCloud2    base 系点云 (步骤4输出)
   /odometry/filtered       nav_msgs/Odometry          融合里程计 (步骤6输出)
   /lidar_odom              nav_msgs/Odometry          LiDAR 里程计 (回退方案)

 发布:
   /map                     nav_msgs/OccupancyGrid     2D 占据栅格地图
   /map_metadata            nav_msgs/MapMetaData       地图元数据
   /current_floor           std_msgs/Int8              当前楼层编号

 发布 TF:
   world → odom             identity (初始), 回环检测后更新

 参数:
   ~map_resolution           float  地图分辨率 (m/格), 默认 0.05
   ~map_width                float  地图宽度 (m), 默认 40.0
   ~map_height               float  地图高度 (m), 默认 50.0
   ~z_min                    float  点云高度下限 (m), 默认 0.1
   ~z_max                    float  点云高度上限 (m), 默认 1.5
   ~prob_hit                 float  击中占据概率, 默认 0.75
   ~prob_miss                float  穿过自由概率, 默认 0.35
   ~min_occupancy            float  最小占据概率, 默认 0.05
   ~max_occupancy            float  最大占据概率, 默认 0.95
   ~update_rate              float  地图发布频率 (Hz), 默认 5.0
   ~floor_count              int    楼层数量, 默认 3
   ~floor_height             float  层高 (m), 默认 3.0
   ~multi_floor_enabled      bool   是否启用多楼层, 默认 true

==============================================================================
"""

import math
import numpy as np
import rospy
import tf2_ros
from sensor_msgs.msg import PointCloud2
from sensor_msgs.point_cloud2 import read_points
from nav_msgs.msg import Odometry, OccupancyGrid, MapMetaData
from geometry_msgs.msg import TransformStamped, Pose, Point, Quaternion, PoseStamped
from std_msgs.msg import Int8, Header
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from tf2_msgs.msg import TFMessage


class GridMapNode:
    """2D 占据栅格地图 + 多楼层管理节点"""

    def __init__(self):
        rospy.init_node('grid_map_node', anonymous=False)
        self.node_name = rospy.get_name()

        # ================================================================
        # 参数
        # ================================================================
        self.resolution = rospy.get_param('~map_resolution', 0.05)
        self.map_width = rospy.get_param('~map_width', 40.0)
        self.map_height = rospy.get_param('~map_height', 50.0)
        self.z_min = rospy.get_param('~z_min', 0.1)
        self.z_max = rospy.get_param('~z_max', 1.5)
        self.prob_hit = rospy.get_param('~prob_hit', 0.75)
        self.prob_miss = rospy.get_param('~prob_miss', 0.35)
        self.min_occupancy = rospy.get_param('~min_occupancy', 0.05)
        self.max_occupancy = rospy.get_param('~max_occupancy', 0.95)
        self.update_rate = rospy.get_param('~update_rate', 5.0)
        self.floor_count = rospy.get_param('~floor_count', 3)
        self.floor_height = rospy.get_param('~floor_height', 3.0)
        self.multi_floor_enabled = rospy.get_param('~multi_floor_enabled', True)
        self.max_ray_length = rospy.get_param('~max_ray_length', 300)
        self.max_map_points = rospy.get_param('~max_map_points', 2000)

        # 地图尺寸 (栅格数)
        self.width_cells = int(self.map_width / self.resolution)
        self.height_cells = int(self.map_height / self.resolution)

        # 地图原点 (world 坐标系左下角)
        self.origin_x = -self.map_width / 2.0
        self.origin_y = -self.map_height / 2.0

        # log-odds 参数
        self.log_odds_hit = math.log(self.prob_hit / (1.0 - self.prob_hit))
        self.log_odds_miss = math.log(self.prob_miss / (1.0 - self.prob_miss))
        self.log_odds_min = math.log(self.min_occupancy / (1.0 - self.min_occupancy))
        self.log_odds_max = math.log(self.max_occupancy / (1.0 - self.max_occupancy))

        # ================================================================
        # 每楼层独立地图 (log-odds 存储)
        # maps[floor_idx] = numpy.ndarray (height_cells, width_cells), float64
        # 初始化为 0 (log-odds 0 = 概率 0.5 = unknown)
        # ================================================================
        self.maps = {}
        for floor in range(self.floor_count):
            self.maps[floor] = np.zeros(
                (self.height_cells, self.width_cells), dtype=np.float64)

        self.current_floor = 0  # 0-based: 0=1F, 1=2F, 2=3F

        # ================================================================
        # 机器人位姿 (world 坐标系)
        # ================================================================
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_z = 0.0
        self.robot_roll = 0.0
        self.robot_pitch = 0.0
        self.robot_yaw = 0.0
        # FAST_LIO 里程计位姿
        self.fast_lio_x = 0.0
        self.fast_lio_y = 0.0
        self.fast_lio_z = 0.0
        self.fast_lio_yaw = 0.0
        # 位姿源优先级: FAST_LIO 新鲜则优先, 否则用 ICP (odom_callback)
        self.last_fast_lio_time = rospy.Time(0)
        self.odom_source_timeout = 1.0  # FAST_LIO 超过该时长未更新则回退 ICP (s)
        self.pose_initialized = False

        # world → odom 回环修正量 (loop_closure_node 经 /loop_closure/correction 下发, 累积值)
        self.world_odom_x = 0.0
        self.world_odom_y = 0.0
        self.world_odom_z = 0.0
        self.world_odom_yaw = 0.0
        # 机器人 world 系位置 (应用回环修正后), 供 ray tracing 起点
        self.robot_world_x = 0.0
        self.robot_world_y = 0.0

        # 上一帧发布地图的时间
        self.last_map_publish_time = rospy.Time.now()

        # ================================================================
        # 订阅
        # ================================================================
        rospy.Subscriber('/cloud_base', PointCloud2,
                         self.cloud_callback, queue_size=10)

        # 使用 FAST_LIO 里程计 (主要) — 替代 ICP, 精度大幅提升
        # 注意: 订阅桥接后的 /fast_lio_odom_base (已航向对齐到 world 系),
        # 而非 raw /fast_lio_odom (camera_init 系, yaw≈0, 与 EKF 差 90°)
        rospy.Subscriber('/fast_lio_odom_base', Odometry,
                         self.fast_lio_callback, queue_size=10)
        # LiDAR 里程计 (ICP, 回退方案)
        rospy.Subscriber('/lidar_odom', Odometry,
                         self.odom_callback, queue_size=10)
        # 回环修正量 (loop_closure_node 发布)
        rospy.Subscriber('/loop_closure/correction', PoseStamped,
                         self.correction_callback, queue_size=5)

        # ================================================================
        # 发布
        # ================================================================
        self.map_pub = rospy.Publisher('/map', OccupancyGrid,
                                        queue_size=5, latch=True)
        self.metadata_pub = rospy.Publisher('/map_metadata', MapMetaData,
                                             queue_size=5, latch=True)
        self.floor_pub = rospy.Publisher('/current_floor', Int8,
                                          queue_size=5, latch=True)

        # ================================================================
        # TF
        # ================================================================
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        # 直接用 /tf_static publisher 确保 world → odom 可靠发布
        self.tf_static_pub = rospy.Publisher('/tf_static', TFMessage,
                                             queue_size=10, latch=True)
        # 发布 world → odom identity TF (初始)
        self._broadcast_world_odom_tf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        # 定时器: 定期发布地图 (降低发布频率节省带宽)
        rospy.Timer(rospy.Duration(1.0 / self.update_rate), self._timer_callback)

        rospy.loginfo("[%s] 栅格地图节点启动完成", self.node_name)
        rospy.loginfo("[%s]   地图尺寸: %.1fm × %.1fm (%.0f × %.0f cells @ %.2fm)",
                      self.node_name, self.map_width, self.map_height,
                      self.width_cells, self.height_cells, self.resolution)
        rospy.loginfo("[%s]   高度过滤: z ∈ [%.1f, %.1f] m", self.node_name,
                      self.z_min, self.z_max)
        rospy.loginfo("[%s]   占据概率: hit=%.2f, miss=%.2f",
                      self.node_name, self.prob_hit, self.prob_miss)
        rospy.loginfo("[%s]   多楼层: %s (层高=%.1fm, %d层)",
                      self.node_name, self.multi_floor_enabled,
                      self.floor_height, self.floor_count)

    # ================================================================
    # 里程计回调
    # ================================================================
    def odom_callback(self, msg):
        """接收里程计数据，更新机器人位姿"""
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        pz = msg.pose.pose.position.z

        # NaN 保护: EKF 或传感器可能发布异常数据
        if math.isnan(px) or math.isnan(py) or math.isnan(pz):
            rospy.logwarn_throttle(5.0, "[%s] 收到 NaN 位姿, 跳过", self.node_name)
            return

        self.robot_x = px
        self.robot_y = py
        self.robot_z = pz

        q = msg.pose.pose.orientation
        self.robot_roll, self.robot_pitch, self.robot_yaw = \
            euler_from_quaternion([q.x, q.y, q.z, q.w])

        if not self.pose_initialized:
            self.pose_initialized = True
            rospy.loginfo("[%s] 位姿已初始化: pos=(%.2f, %.2f, %.2f), yaw=%.2f°",
                          self.node_name, self.robot_x, self.robot_y,
                          self.robot_z, math.degrees(self.robot_yaw))

    def fast_lio_callback(self, msg):
        """接收 FAST_LIO 里程计 (主定位源, 6-DoF 含 z 供楼层判断)
        订阅 /fast_lio_odom_base: 已经 fastlio_bridge 航向对齐到 world 系
        (frame_id=odom, child_frame_id=base), 与 EKF 输出一致
        只存 fast_lio_* 并记时间戳, 不直接写 robot_*, 由 _refresh_pose 统一解析
        """
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        pz = msg.pose.pose.position.z

        if math.isnan(px) or math.isnan(py) or math.isnan(pz):
            return

        self.fast_lio_x = px
        self.fast_lio_y = py
        self.fast_lio_z = pz

        q = msg.pose.pose.orientation
        _, _, fl_yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.fast_lio_yaw = fl_yaw

        self.last_fast_lio_time = msg.header.stamp

        if not self.pose_initialized:
            self.pose_initialized = True
            rospy.loginfo("[%s] FAST-LIO 位姿已初始化: pos=(%.2f, %.2f, %.2f), yaw=%.2f°",
                          self.node_name, self.fast_lio_x, self.fast_lio_y,
                          self.fast_lio_z, math.degrees(self.fast_lio_yaw))

    # ================================================================
    # 位姿源优先级解析
    # ================================================================
    def _refresh_pose(self):
        """解析当前使用的位姿源: FAST_LIO 新鲜则优先, 否则用 ICP 里程计。
        统一在此处把选定位姿写入 self.robot_*, 并做楼层检测,
        避免 FAST_LIO 与 ICP 两个回调并发写 robot_* 造成位姿抖动。
        """
        now = rospy.Time.now()
        if self.last_fast_lio_time != rospy.Time(0) and \
                (now - self.last_fast_lio_time).to_sec() < self.odom_source_timeout:
            self.robot_x = self.fast_lio_x
            self.robot_y = self.fast_lio_y
            self.robot_z = self.fast_lio_z
            self.robot_yaw = self.fast_lio_yaw

        # world 系机器人位置 = T_world_odom · T_odom_base · (0,0,0)
        #   = R(world_odom_yaw)·[robot_x, robot_y] + [world_odom_x, world_odom_y]
        cos_c = math.cos(self.world_odom_yaw)
        sin_c = math.sin(self.world_odom_yaw)
        self.robot_world_x = cos_c * self.robot_x - sin_c * self.robot_y + self.world_odom_x
        self.robot_world_y = sin_c * self.robot_x + cos_c * self.robot_y + self.world_odom_y

        # 楼层检测统一在这里做 (robot_z 已被解析为当前源)
        if self.multi_floor_enabled:
            self._update_floor()

    # ================================================================
    # 回环修正量回调
    # ================================================================
    def correction_callback(self, msg):
        """接收回环修正量 (loop_closure_node 发布, 累积值 world→odom)"""
        self.world_odom_x = msg.pose.position.x
        self.world_odom_y = msg.pose.position.y
        self.world_odom_z = msg.pose.position.z
        q = msg.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.world_odom_yaw = yaw
        rospy.loginfo("[%s] 收到回环修正: (%.3f, %.3f, %.3f) yaw=%.2f°",
                      self.node_name, self.world_odom_x, self.world_odom_y,
                      self.world_odom_z, math.degrees(self.world_odom_yaw))

    # ================================================================
    # 点云回调 (主流水线)
    # ================================================================
    def cloud_callback(self, msg):
        """
        每帧点云到达时更新地图:
         1. 解析点云
         2. 高度过滤
         3. 变换到 world 坐标系
         4. log-odds 占据概率更新
        """
        if not self.pose_initialized:
            return

        # 位姿源优先级解析 (FAST_LIO 优先, ICP 回退) + 楼层检测
        self._refresh_pose()

        # ---- 1. 解析点云 ----
        points_arr = self._pointcloud2_to_numpy(msg)
        if points_arr is None or len(points_arr) == 0:
            return

        n_raw = len(points_arr)

        # ---- 1.5 下采样 (CPU优化: 限制建图点数) ----
        if len(points_arr) > self.max_map_points:
            idx = np.random.choice(len(points_arr), self.max_map_points,
                                   replace=False)
            points_arr = points_arr[idx]

        # ---- 2. 高度过滤 (base 系 z, 相对机器人) ----
        # 点云仍在 base 系, 墙体相对机器人的高度恒为 [z_min, z_max],
        # 不应加楼层偏移 (否则 2F/3F 的 base_z 只有 ~0~1.5, 会被全部滤掉,
        # 导致高楼层地图永远为空)。
        z_vals = points_arr[:, 2]
        mask = (z_vals >= self.z_min) & (z_vals <= self.z_max) & \
               (~np.isnan(z_vals))
        points_arr = points_arr[mask]
        n_after_filter = len(points_arr)
        if n_after_filter == 0:
            rospy.logwarn_throttle(10.0,
                "[%s] 高度过滤后0点! raw=%d z∈[%.2f,%.2f] z_sample=[%.2f,%.2f,%.2f]",
                self.node_name, n_raw, self.z_min, self.z_max,
                float(z_vals[0]) if len(z_vals) > 0 else 0.0,
                float(z_vals[len(z_vals)//2]) if len(z_vals) > 0 else 0.0,
                float(z_vals[-1]) if len(z_vals) > 0 else 0.0)
            return

        # ---- 3. 变换到 world 坐标系 (向量化, 完整链 base → odom → world) ----
        # P_odom  = R(robot_yaw)·P_base + [robot_x, robot_y]
        # P_world = R(world_odom_yaw)·P_odom + [world_odom_x, world_odom_y]
        cos_y = math.cos(self.robot_yaw)
        sin_y = math.sin(self.robot_yaw)
        cos_c = math.cos(self.world_odom_yaw)
        sin_c = math.sin(self.world_odom_yaw)

        px, py = points_arr[:, 0].copy(), points_arr[:, 1].copy()
        x_odom = cos_y * px - sin_y * py + self.robot_x
        y_odom = sin_y * px + cos_y * py + self.robot_y
        points_arr[:, 0] = cos_c * x_odom - sin_c * y_odom + self.world_odom_x
        points_arr[:, 1] = sin_c * x_odom + cos_c * y_odom + self.world_odom_y
        points_arr[:, 2] = points_arr[:, 2] + self.robot_z + self.world_odom_z

        # ---- 4. 占据概率更新 ----
        self._update_grid(points_arr)

        # 调试: 统计非 -1 栅格数
        grid = self.maps[self.current_floor]
        n_known = int(np.sum(np.abs(grid) > 1e-9))
        rospy.loginfo_throttle(5.0,
            "[%s] raw=%d filtered=%d known_cells=%d pos=(%.2f,%.2f) yaw=%.1f°",
            self.node_name, n_raw, n_after_filter, n_known,
            self.robot_x, self.robot_y, math.degrees(self.robot_yaw))

    def _update_grid(self, points):
        """
        log-odds 占据概率更新

        对每个点云点:
          - hit:  被激光击中的栅格，占据概率增加
          - miss:  机器人到点之间的所有栅格，被激光穿过，占据概率降低

        使用 Bresenham 光线追踪模拟 miss 路径。
        简化: 对每个 hit 栅格做 log_odds_hit 增加；
              对机器人所在栅格到 hit 栅格之间的所有栅格做 log_odds_miss 减少。

        只更新当前楼层的地图。
        """
        grid = self.maps[self.current_floor]

        # 机器人所在栅格 (world 系, 应用回环修正后)
        robot_cx = int((self.robot_world_x - self.origin_x) / self.resolution)
        robot_cy = int((self.robot_world_y - self.origin_y) / self.resolution)

        max_ray = self.max_ray_length

        for pt in points:
            px, py = pt[0], pt[1]

            cx = int((px - self.origin_x) / self.resolution)
            cy = int((py - self.origin_y) / self.resolution)

            if not (0 <= cx < self.width_cells and 0 <= cy < self.height_cells):
                continue

            # --- hit ---
            grid[cy, cx] = min(grid[cy, cx] + self.log_odds_hit,
                               self.log_odds_max)

            # --- miss: Bresenham (限制最长光线减少CPU) ---
            if not (0 <= robot_cx < self.width_cells and
                    0 <= robot_cy < self.height_cells):
                continue

            # 跳过过远的光线 (> max_ray 格)
            if abs(cx - robot_cx) > max_ray or abs(cy - robot_cy) > max_ray:
                continue

            step = 0
            for mx, my in self._bresenham(robot_cx, robot_cy, cx, cy):
                if step >= max_ray:
                    break
                if 0 <= mx < self.width_cells and 0 <= my < self.height_cells:
                    new_val = grid[my, mx] + self.log_odds_miss
                    grid[my, mx] = max(min(new_val, self.log_odds_max),
                                       self.log_odds_min)
                step += 1

    # ================================================================
    # Bresenham 光线追踪
    # ================================================================
    def _bresenham(self, x0, y0, x1, y1):
        """
        Bresenham 线段算法，返回从 (x0,y0) 到 (x1,y1) 线段
        所经过的所有栅格 (不包括终点，因为终点是 hit)
        """
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        x, y = x0, y0
        max_steps = dx + dy + 10  # 安全上限
        steps = 0

        while steps < max_steps:
            if x == x1 and y == y1:
                break
            points.append((x, y))
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
            steps += 1

        return points

    # ================================================================
    # 楼层管理 (Step 8)
    # ================================================================
    def _update_floor(self):
        """根据机器人 z 坐标检测并切换当前楼层"""
        new_floor = int(self.robot_z / self.floor_height)
        new_floor = max(0, min(new_floor, self.floor_count - 1))

        if new_floor != self.current_floor:
            rospy.loginfo("[%s] 楼层切换: %dF → %dF (z=%.2f m)",
                          self.node_name, self.current_floor + 1,
                          new_floor + 1, self.robot_z)
            self.current_floor = new_floor
            # 发布楼层切换通知
            floor_msg = Int8()
            floor_msg.data = self.current_floor
            self.floor_pub.publish(floor_msg)

    # ================================================================
    # 定时发布地图
    # ================================================================
    def _timer_callback(self, event):
        """定时发布当前楼层的地图"""
        now = rospy.Time.now()

        # 持续广播 world → odom TF (含回环修正量)
        self._broadcast_world_odom_tf(self.world_odom_x, self.world_odom_y,
                                      self.world_odom_z, 0.0, 0.0,
                                      self.world_odom_yaw)

        grid = self.maps[self.current_floor]

        # log-odds → OccupancyGrid 值 [0, 100], -1=unknown
        # probability = 1 - 1/(1 + exp(log_odds))
        # occupancy = probability * 100
        occupancy_data = np.zeros(
            self.height_cells * self.width_cells, dtype=np.int8)

        for cy in range(self.height_cells):
            for cx in range(self.width_cells):
                log_odd = grid[cy, cx]
                if abs(log_odd) < 1e-9:
                    occupancy_data[cy * self.width_cells + cx] = -1  # unknown
                else:
                    # Clamp log_odd to prevent math.exp() overflow (exp(709) ≈ 8e307)
                    log_odd_clamped = max(min(log_odd, 100.0), -100.0)
                    prob = 1.0 - 1.0 / (1.0 + math.exp(log_odd_clamped))
                    val = int(prob * 100)
                    val = max(0, min(100, val))
                    occupancy_data[cy * self.width_cells + cx] = val

        # 构建 OccupancyGrid 消息
        grid_msg = OccupancyGrid()
        grid_msg.header.stamp = now
        grid_msg.header.frame_id = 'world'

        grid_msg.info = MapMetaData()
        grid_msg.info.resolution = self.resolution
        grid_msg.info.width = self.width_cells
        grid_msg.info.height = self.height_cells
        grid_msg.info.origin = Pose()
        grid_msg.info.origin.position.x = self.origin_x
        grid_msg.info.origin.position.y = self.origin_y
        grid_msg.info.origin.position.z = 0.0
        grid_msg.info.origin.orientation.w = 1.0

        grid_msg.data = list(occupancy_data)

        self.map_pub.publish(grid_msg)
        self.metadata_pub.publish(grid_msg.info)

        self.last_map_publish_time = now

    # ================================================================
    # TF 广播: world → odom (建图接管)
    # ================================================================
    def _broadcast_world_odom_tf(self, x, y, z, roll, pitch, yaw):
        """广播 world → odom TF (建图 pipeline 统一管理)"""
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = 'world'
        t.child_frame_id = 'odom'

        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = z

        q = quaternion_from_euler(roll, pitch, yaw)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        # 同时发 /tf 和 /tf_static (确保可靠到达)
        self.tf_broadcaster.sendTransform(t)
        tfm = TFMessage([t])
        self.tf_static_pub.publish(tfm)

    # ================================================================
    # 点云解析: PointCloud2 → numpy
    # ================================================================
    def _pointcloud2_to_numpy(self, msg):
        """解析 PointCloud2 为 numpy 数组"""
        gen = read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)
        points_list = list(gen)
        if len(points_list) == 0:
            return None
        return np.array(points_list, dtype=np.float32)


# ====================================================================
def main():
    try:
        node = GridMapNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("[grid_map_node] 节点正常退出")
    except KeyboardInterrupt:
        rospy.loginfo("[grid_map_node] 收到 Ctrl+C, 退出")


if __name__ == '__main__':
    main()
