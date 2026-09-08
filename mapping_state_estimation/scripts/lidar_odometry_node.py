#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
  LiDAR 里程计节点 (Step 5)
==============================================================================
 功能: 订阅 /cloud_base 处理后的点云，通过连续帧 ICP 扫描匹配估计机器人运动，
       累积得到 LiDAR 里程计 (6-DoF 位姿)。

 订阅话题:
   /cloud_base          sensor_msgs/PointCloud2    base 坐标系下的干净点云 (10 Hz)
   /imu/attitude        geometry_msgs/Vector3Stamped  IMU 姿态 (roll/pitch/yaw)

 发布话题:
   /lidar_odom          nav_msgs/Odometry          LiDAR 里程计 (odom 坐标系下)
   /lidar_odom_path     nav_msgs/Path              轨迹路径 (用于 RViz 可视化)
   /cloud_keyframe      sensor_msgs/PointCloud2    关键帧点云 (用于回环检测/全局优化)

 发布 TF:
   odom → base          (基于 LiDAR 里程计的位姿变换)

 参数:
   ~icp_max_iterations                 int    ICP 最大迭代次数, 默认 30
   ~icp_max_correspondence_dist        float  最大对应点距离 (m), 默认 1.0
   ~icp_transformation_epsilon         float  变换收敛阈值, 默认 1e-5
   ~icp_euclidean_fitness_epsilon      float  适应度收敛阈值, 默认 1e-5
   ~keyframe_translation_thresh        float  关键帧最小平移距离 (m), 默认 0.5
   ~keyframe_rotation_thresh           float  关键帧最小旋转角度 (rad), 默认 0.2
   ~use_imu_prediction                 bool   使用 IMU 姿态作为 ICP 初始猜测, 默认 true
   ~publish_tf                         bool   是否发布 odom→base TF, 默认 true
   ~odom_frame                         str    odom 坐标系名称, 默认 "odom"
   ~base_frame                         str    base 坐标系名称, 默认 "base"

==============================================================================
 ICP 算法说明
==============================================================================

 使用的 ICP 变体: 点到点 SVD ICP (Point-to-Point)

 目标: 找到刚性变换 T = (R, t)，使得源点云 A 与目标点云 B 对齐:
        min Σ ||R * a_i + t - b_j||²

 每轮迭代:
   1. 最近邻关联 (Correspondence):
      对源点云每个点 a_i，在目标点云 B 中找最近邻 b_j
      仅保留距离 < max_correspondence_dist 的对应对
   2. 变换估计 (SVD):
      计算两组点的中心 c_A, c_B
      构建互协方差矩阵 H = Σ (a_i - c_A) * (b_j - c_B)^T
      对 H 做 SVD: H = U · Σ · V^T
      最优旋转: R = V · U^T  (处理反射情况)
      最优平移: t = c_B - R · c_A
   3. 更新源点云: a_i' = R · a_i + t
   4. 检查收敛: |Δtranslation| < ε 且 |Δrotation| < ε

 为加速最近邻搜索，使用分块 numpy 向量化计算（避免 scipy 依赖）。

==============================================================================
 坐标系说明
==============================================================================

 /cloud_base 的点已经在 odom 坐标系下 (由 pointcloud2livox.py 变换)。
 本节点的 ICP 在 odom 坐标系下直接配准两帧点云，得到的 frame-to-frame
 变换就是机器人在 odom 坐标系下的增量运动。

 里程计位姿链:
   odom → base
   其中 base 由累积的 ICP 增量变换得到，odom 为世界固定坐标系。

==============================================================================
"""

import math
import time
import numpy as np
import rospy
import tf2_ros
import tf
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs.point_cloud2 import read_points, create_cloud
from geometry_msgs.msg import Vector3Stamped, TransformStamped, Quaternion
from geometry_msgs.msg import PoseStamped, Pose
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Header
from tf.transformations import quaternion_from_euler, euler_from_quaternion
# scipy 未安装，使用纯 numpy 实现最近邻搜索，见 _nearest_neighbors()


class LidarOdometryNode:
    """LiDAR 里程计节点

    接收处理后的点云和 IMU 姿态，通过 ICP 帧间匹配估计机器人运动。
    """

    def __init__(self):
        # ============================================================
        # 1. ROS 节点初始化
        # ============================================================
        rospy.init_node('lidar_odometry_node', anonymous=False)
        self.node_name = rospy.get_name()

        # ============================================================
        # 2. 读取参数
        # ============================================================
        self.icp_max_iter = int(float(rospy.get_param('~icp_max_iterations', 20)))
        self.icp_max_dist = float(rospy.get_param('~icp_max_correspondence_dist', 1.0))
        self.icp_trans_eps = float(rospy.get_param('~icp_transformation_epsilon', 1e-5))
        self.icp_euclidean_eps = float(rospy.get_param('~icp_euclidean_fitness_epsilon', 1e-5))
        self.max_cloud_points = int(float(rospy.get_param('~max_cloud_points', 3000)))

        self.keyframe_trans_thresh = float(rospy.get_param('~keyframe_translation_thresh', 0.5))
        self.keyframe_rot_thresh = float(rospy.get_param('~keyframe_rotation_thresh', 0.2))

        self.use_imu_prediction = str(rospy.get_param('~use_imu_prediction', True)).lower() in ('true', '1')
        self.publish_tf = str(rospy.get_param('~publish_tf', True)).lower() in ('true', '1')
        self.odom_frame = rospy.get_param('~odom_frame', 'odom')
        self.base_frame = rospy.get_param('~base_frame', 'base')

        rospy.loginfo("[%s] ICP 参数:", self.node_name)
        rospy.loginfo("[%s]   最大迭代: %d", self.node_name, self.icp_max_iter)
        rospy.loginfo("[%s]   最大对应距离: %.2f m", self.node_name, self.icp_max_dist)
        rospy.loginfo("[%s]   变换收敛阈值: %.1e", self.node_name, self.icp_trans_eps)
        rospy.loginfo("[%s]   关键帧平移阈值: %.2f m", self.node_name, self.keyframe_trans_thresh)
        rospy.loginfo("[%s]   关键帧旋转阈值: %.2f rad", self.node_name, self.keyframe_rot_thresh)
        rospy.loginfo("[%s]   使用 IMU 预测: %s", self.node_name, self.use_imu_prediction)

        # ============================================================
        # 3. 状态变量
        # ============================================================
        # 当前累积位姿 (odom 坐标系下 base 的位姿)
        #   position: (x, y, z)
        #   orientation: (roll, pitch, yaw)  [弧度]
        self.cur_x = 0.0
        self.cur_y = 0.0
        self.cur_z = 0.0
        self.cur_roll = 0.0
        self.cur_pitch = 0.0
        self.cur_yaw = 0.0

        # 上一帧点云 (目标点云, 即 "地图" 坐标系下的点)
        self.prev_cloud = None

        # 上一帧关键帧位姿
        self.last_keyframe_x = 0.0
        self.last_keyframe_y = 0.0
        self.last_keyframe_z = 0.0
        self.last_keyframe_roll = 0.0
        self.last_keyframe_pitch = 0.0
        self.last_keyframe_yaw = 0.0
        self.last_keyframe_cloud = None

        # IMU 姿态 (用于 ICP 初始猜测)
        self.latest_imu_roll = 0.0
        self.latest_imu_pitch = 0.0
        self.latest_imu_yaw = 0.0
        self.prev_imu_yaw = 0.0
        self.imu_initialized = False

        # 统计
        self.frame_count = 0
        self.icp_fail_count = 0
        self.icp_total_time = 0.0

        # ============================================================
        # 4. 订阅者
        # ============================================================
        rospy.Subscriber('/cloud_base', PointCloud2,
                         self.cloud_callback, queue_size=10)
        rospy.Subscriber('/imu/attitude', Vector3Stamped,
                         self.imu_attitude_callback, queue_size=50)

        # ============================================================
        # 5. 发布者
        # ============================================================
        self.odom_pub = rospy.Publisher('/lidar_odom', Odometry, queue_size=10)
        self.path_pub = rospy.Publisher('/lidar_odom_path', Path, queue_size=10)
        self.keyframe_pub = rospy.Publisher('/cloud_keyframe', PointCloud2, queue_size=10)

        # 轨迹
        self.path = Path()
        self.path.header.frame_id = self.odom_frame

        # ============================================================
        # 6. TF2 广播器
        # ============================================================
        if self.publish_tf:
            self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        rospy.loginfo("[%s] LiDAR 里程计节点启动完成", self.node_name)
        rospy.loginfo("[%s] 订阅: /cloud_base, /imu/attitude", self.node_name)
        rospy.loginfo("[%s] 发布: /lidar_odom, /lidar_odom_path, /cloud_keyframe",
                      self.node_name)

    # ================================================================
    # IMU 姿态回调
    # ================================================================
    def imu_attitude_callback(self, msg):
        """
        接收 /imu/attitude (Vector3Stamped)
          vector.x = roll  [rad]
          vector.y = pitch [rad]
          vector.z = yaw   [rad]
        """
        self.latest_imu_roll = msg.vector.x
        self.latest_imu_pitch = msg.vector.y
        self.latest_imu_yaw = msg.vector.z
        self.imu_initialized = True

    # ================================================================
    # 点云回调 (主处理流水线)
    # ================================================================
    def cloud_callback(self, msg):
        """
        主处理流水线:
          当前帧点云 → (可选: IMU 初始猜测) → ICP 配准 → 累积位姿 → 发布里程计
        """
        # ---- 0. 解析点云为 numpy 数组 ----
        cloud_cur, has_intensity = self._pointcloud2_to_numpy(msg)
        if cloud_cur is None or len(cloud_cur) < 50:
            rospy.logwarn_throttle(5.0,
                                   "[%s] 当前帧点数不足 (%d), 跳过",
                                   self.node_name,
                                   len(cloud_cur) if cloud_cur is not None else 0)
            return

        # ---- 0.5 限制点云大小 (CPU优化: 超过上限则随机下采样) ----
        if len(cloud_cur) > self.max_cloud_points:
            idx = np.random.choice(len(cloud_cur), self.max_cloud_points,
                                   replace=False)
            cloud_cur = cloud_cur[idx]

        self.frame_count += 1
        stamp = msg.header.stamp

        # ---- 1. 第一帧: 初始化 ----
        if self.prev_cloud is None:
            self.prev_cloud = cloud_cur
            self.prev_imu_yaw = self.latest_imu_yaw
            # 用 IMU 初始 yaw 对齐 odom 系 (同在 Gazebo world 系)
            if self.imu_initialized:
                self.cur_yaw = self.latest_imu_yaw
                rospy.loginfo("[%s] 初始化 yaw=%.1f° (IMU)",
                              self.node_name, math.degrees(self.cur_yaw))
            self._publish_all(stamp)
            rospy.loginfo("[%s] 初始化完成，等待下一帧...", self.node_name)
            return

        # ---- 2. 计算 ICP 初始猜测 ----
        init_guess = self._compute_initial_guess()

        # ---- 3. ICP 配准 ----
        t_start = time.time()
        T, converged, fitness, inlier_ratio = self._icp_svd(
            source=cloud_cur,          # 当前帧 (源)
            target=self.prev_cloud,    # 上一帧 (目标)
            init_guess=init_guess,
            max_iterations=self.icp_max_iter,
            max_correspondence_dist=self.icp_max_dist,
            transformation_epsilon=self.icp_trans_eps,
            euclidean_fitness_epsilon=self.icp_euclidean_eps,
        )
        t_elapsed = time.time() - t_start
        self.icp_total_time += t_elapsed

        if not converged:
            self.icp_fail_count += 1
            rospy.logwarn_throttle(5.0,
                                   "[%s] ICP 未收敛 (fitness=%.4f, inliers=%.1f%%)",
                                   self.node_name, fitness, inlier_ratio * 100)
            # 即使未收敛也使用结果，避免里程计断裂
            # 但保留上一帧点云不变，不更新

        # ---- 4. 提取增量变换 ----
        # T 是 source(当前帧) 到 target(上一帧) 的刚性变换
        # 上一帧的点云在 odom 坐标系下 (即历史时刻的 base 坐标系)
        # 当前帧的点云也已在 odom 坐标系下 (由 pointcloud2livox.py 转换)
        #
        # 因此 T 表示: 从 "当前 base 位姿" 到 "上一帧 base 位姿" 的修正
        # 即: p_target = T · p_source
        #
        # 累积位姿的方式:
        #   设 T_world←prev 为上一帧 base 在世界(odom)坐标系的位姿
        #   设 T_prev←cur 为 ICP 的到的变换 (source → target)
        #   则当前位姿: T_world←cur = T_world←prev · T_prev←cur
        #   等价于先应用 ICP 增量，再累积到世界坐标系
        #
        #   T_world←cur = T_world←prev · T

        # ---- 4. 提取增量变换 ----
        # ICP 计算: T_prev←cur = [R | t], 即 p_prev = R * p_cur + t
        # 点云都在 base 系: source=base_cur, target=base_prev
        #
        # 位姿累积: T_world←cur = T_world←prev · T_prev←cur
        #   R_world←cur = R_world←prev · R
        #   pos_cur = pos_prev + R_world←prev · t
        #
        # 所以:
        #   机器人旋转(相对prev): dR = R (NOT R.T)
        #   机器人平移(body系):   dt_body = t (NOT -R.T@t)

        R = T[:3, :3]
        t_vec = T[:3, 3]
        _, _, dyaw_icp = self._rotation_matrix_to_euler(R)

        # ---- 5. 累积位姿 (2D约束: 地面机器人只用 x,y,yaw) ----
        prev_yaw = self.cur_yaw  # 本帧开始前的 yaw

        # IMU yaw 替换: Mid-360非重复扫描+四足震荡导致ICP虚假yaw漂移
        if self.imu_initialized:
            dyaw_imu = self.latest_imu_yaw - self.prev_imu_yaw
            dyaw_imu = math.atan2(math.sin(dyaw_imu), math.cos(dyaw_imu))
            self.cur_yaw += dyaw_imu
        else:
            self.cur_yaw += dyaw_icp
        # 2D约束: 不累积 ICP 的 droll/dpitch (四足震荡导致万向节死锁, yaw翻转180°)
        # self.cur_roll += droll   # 禁用以防 gimbal lock
        # self.cur_pitch += dpitch
        self.cur_z = 0.0  # 2D约束: z 固定为0

        # dt_body = t_vec: 机器人在 body(prev) 系下的平移增量 (仅用 xy, 忽略 z)
        # 旋转到 odom/world 系: dt_world = R(prev_yaw) · dt_body
        cy = math.cos(prev_yaw)
        sy = math.sin(prev_yaw)

        self.cur_x += cy * t_vec[0] - sy * t_vec[1]
        self.cur_y += sy * t_vec[0] + cy * t_vec[1]
        # self.cur_z += t_vec[2]   # 2D约束: 不累积 z 平移

        # 角度归一化
        self.cur_yaw = math.atan2(math.sin(self.cur_yaw),
                                  math.cos(self.cur_yaw))

        # ---- 6. 关键帧管理 & 更新 ICP 目标 ----
        # Mid-360 非重复扫描: 相邻帧激光线完全不同，
        # 只有关键帧时机才更新 ICP target，保证 target 有稳定几何结构
        is_keyframe = self._check_keyframe(cloud_cur, stamp)
        if is_keyframe:
            self.prev_cloud = cloud_cur
        self.prev_imu_yaw = self.latest_imu_yaw

        # ---- 7. 发布 ----
        self._publish_all(stamp)

    # ================================================================
    # ICP 初始猜测
    # ================================================================
    def _compute_initial_guess(self):
        """
        基于 IMU 姿态变化计算 ICP 初始猜测

        原理:
          如果 IMU 的 yaw 从上一帧到当前帧变化了 Δyaw，
          说明机器人旋转了。预估一个初始旋转矩阵给 ICP，加快收敛。

        Returns:
            4x4 刚体变换矩阵 (初始猜测), 无先验时为 None
        """
        if not self.use_imu_prediction or not self.imu_initialized:
            return None

        dyaw = self.latest_imu_yaw - self.prev_imu_yaw
        dyaw = math.atan2(math.sin(dyaw), math.cos(dyaw))

        # 如果 yaw 变化很小，不做预测
        if abs(dyaw) < 0.005:  # ~0.3°
            return None

        # 构造初始猜测: 纯 Z 轴旋转 dyaw
        cos_y = math.cos(dyaw)
        sin_y = math.sin(dyaw)

        T_guess = np.eye(4, dtype=np.float64)
        T_guess[0, 0] = cos_y
        T_guess[0, 1] = -sin_y
        T_guess[1, 0] = sin_y
        T_guess[1, 1] = cos_y

        return T_guess

    # ================================================================
    # ICP: SVD 点到点 ICP
    # ================================================================
    def _icp_svd(self, source, target, init_guess=None,
                 max_iterations=30,
                 max_correspondence_dist=1.0,
                 transformation_epsilon=1e-5,
                 euclidean_fitness_epsilon=1e-5):
        """
        迭代最近点 (ICP) 算法 — 点到点 SVD 变体

        参数:
          source: (N, 3+) numpy 当前帧点云 (前3列为 xyz)
          target: (M, 3+) numpy 上一帧点云
          init_guess: (4, 4) 初始变换矩阵, None 表示用单位矩阵
          max_iterations: 最大迭代次数
          max_correspondence_dist: 对应点最大距离 (m)
          transformation_epsilon: 变换收敛阈值
          euclidean_fitness_epsilon: 适应度收敛阈值

        返回:
          T: (4, 4) 最终变换矩阵
          converged: bool 是否收敛
          fitness: float MSE 误差
          inlier_ratio: float 内点比例
        """
        src = source[:, :3].copy().astype(np.float64)
        tgt = target[:, :3].copy().astype(np.float64)

        n_src = len(src)

        # 初始化累积变换
        T = np.eye(4, dtype=np.float64)

        # 应用初始猜测
        if init_guess is not None:
            # init_guess 是从 source 到 target 的猜测
            # 预旋转 src: src' = guess @ src
            homog = np.ones((n_src, 4), dtype=np.float64)
            homog[:, :3] = src
            src = (init_guess @ homog.T).T[:, :3]
            T = init_guess.copy()

        prev_error = float('inf')

        for iteration in range(max_iterations):
            # Step A: 最近邻关联
            # 对 src 中每个点找 tgt 中最近邻
            # 纯 numpy 实现，分块控制内存，详见 _nearest_neighbors()
            distances, indices = self._nearest_neighbors(
                src, tgt, max_dist=max_correspondence_dist)

            # 筛选有效对应
            valid_mask = distances < max_correspondence_dist
            n_valid = int(valid_mask.sum())

            if n_valid < 10:
                rospy.logwarn_throttle(5.0,
                                       "[%s] ICP iter %d: 有效对应点不足 (%d < 10)",
                                       self.node_name, iteration, n_valid)
                return T, False, float('inf'), 0.0

            src_valid = src[valid_mask]
            tgt_valid = tgt[indices[valid_mask]]

            # Step B: SVD 求解最优变换
            # 计算中心
            c_src = src_valid.mean(axis=0)
            c_tgt = tgt_valid.mean(axis=0)

            # 去中心化
            src_centered = src_valid - c_src
            tgt_centered = tgt_valid - c_tgt

            # 互协方差矩阵 H = src_centered^T * tgt_centered  (3 × N) × (N × 3) = 3×3
            H = src_centered.T @ tgt_centered  # shape (3, 3)

            # SVD
            U, S, Vt = np.linalg.svd(H)
            V = Vt.T
            Ut = U.T

            # 最优旋转 (处理反射情况)
            det = np.linalg.det(V @ Ut)
            correction = np.eye(3)
            correction[2, 2] = det  # 如果 det < 0，修正为 det=+1
            R_iter = V @ correction @ Ut

            # 最优平移
            t_iter = c_tgt - R_iter @ c_src

            # 构造本轮变换
            T_iter = np.eye(4, dtype=np.float64)
            T_iter[:3, :3] = R_iter
            T_iter[:3, 3] = t_iter

            # Step C: 应用变换
            homog = np.ones((n_src, 4), dtype=np.float64)
            homog[:, :3] = src
            src = (T_iter @ homog.T).T[:, :3]

            # 累积变换: T = T_iter · T
            T = T_iter @ T

            # Step D: 检查收敛
            # 平移变化量
            dt_norm = np.linalg.norm(t_iter)
            # 旋转变化量 (等价旋转角)
            trace_R = np.trace(R_iter)
            cos_angle = (trace_R - 1.0) / 2.0
            cos_angle = max(-1.0, min(1.0, cos_angle))
            rot_angle = abs(math.acos(cos_angle))

            # 适应度 (MSE)
            current_error = float(np.mean(distances[valid_mask] ** 2))

            # 内点比例
            inlier_ratio = n_valid / n_src

            if dt_norm < transformation_epsilon and rot_angle < transformation_epsilon:
                rospy.logdebug("[%s] ICP 收敛于 iter %d: dt=%.6f, angle=%.6f rad",
                               self.node_name, iteration, dt_norm, rot_angle)
                return T, True, current_error, inlier_ratio

            if abs(prev_error - current_error) < euclidean_fitness_epsilon:
                rospy.logdebug("[%s] ICP 适应度收敛于 iter %d: error=%.6f",
                               self.node_name, iteration, current_error)
                return T, True, current_error, inlier_ratio

            prev_error = current_error

        # 达到最大迭代
        return T, True, prev_error, inlier_ratio

    # ================================================================
    # 最近邻搜索: 纯 numpy 实现 (替代 scipy.spatial.cKDTree)
    # ================================================================
    def _nearest_neighbors(self, src, tgt, max_dist, batch_size=500):
        """
        对 src 中每个点，在 tgt 中找最近邻

        纯 numpy 分块实现，不依赖 scipy。时间复杂度 O(|src|×|tgt|)，
        但通过向量化广播，对体素下采样后的点云 (~2000点) 通常 < 5ms。

        参数:
          src: (N, 3) 源点云
          tgt: (M, 3) 目标点云
          max_dist: 最大有效距离，超过此距离的匹配标记为无效
          batch_size: 源点云分块大小，控制峰值内存

        返回:
          distances: (N,) 最近邻距离 (无效匹配填充 max_dist)
          indices:   (N,) 最近邻索引 (无效匹配填充 0)
        """
        n_src = len(src)
        n_tgt = len(tgt)

        all_distances = []
        all_indices = []

        for start in range(0, n_src, batch_size):
            end = min(start + batch_size, n_src)
            batch = src[start:end]  # (B, 3)

            # 向量化距离计算: diff shape (B, M, 3)
            diff = batch[:, None, :] - tgt[None, :, :]
            dist = np.sqrt(np.sum(diff * diff, axis=2))  # (B, M)

            # 每行找最小距离及其索引
            batch_min_idx = np.argmin(dist, axis=1)       # (B,)
            batch_min_dist = dist[np.arange(len(batch)), batch_min_idx]  # (B,)

            all_distances.append(batch_min_dist)
            all_indices.append(batch_min_idx)

        distances = np.concatenate(all_distances)
        indices = np.concatenate(all_indices)

        # 超过 max_dist 的标记为无效
        distances = np.where(distances <= max_dist, distances, max_dist)

        return distances, indices

    # ================================================================
    # 关键帧管理
    # ================================================================
    def _check_keyframe(self, cloud, stamp):
        """
        判断是否需要创建新关键帧

        条件: 当前位姿与上一个关键帧的距离 > 阈值 或 角度差 > 阈值
        """
        # 平移差
        dx = self.cur_x - self.last_keyframe_x
        dy = self.cur_y - self.last_keyframe_y
        dz = self.cur_z - self.last_keyframe_z
        trans_dist = math.sqrt(dx * dx + dy * dy + dz * dz)

        # 角度差 (使用最小角度差)
        dyaw = self.cur_yaw - self.last_keyframe_yaw
        dyaw = abs(math.atan2(math.sin(dyaw), math.cos(dyaw)))

        if trans_dist >= self.keyframe_trans_thresh or dyaw >= self.keyframe_rot_thresh:
            self.last_keyframe_x = self.cur_x
            self.last_keyframe_y = self.cur_y
            self.last_keyframe_z = self.cur_z
            self.last_keyframe_roll = self.cur_roll
            self.last_keyframe_pitch = self.cur_pitch
            self.last_keyframe_yaw = self.cur_yaw
            self.last_keyframe_cloud = cloud.copy()

            # 发布关键帧点云
            keyframe_msg = self._numpy_to_pointcloud2(
                cloud, stamp, self.odom_frame)
            self.keyframe_pub.publish(keyframe_msg)

            rospy.loginfo("[%s] 新关键帧: pos=(%.2f, %.2f, %.2f), yaw=%.2f°",
                          self.node_name,
                          self.cur_x, self.cur_y, self.cur_z,
                          math.degrees(self.cur_yaw))
            return True
        return False

    # ================================================================
    # 发布
    # ================================================================
    def _publish_all(self, stamp):
        """发布里程计、TF 和路径"""
        # -- 里程计消息 --
        odom_msg = self._build_odom_msg(stamp)
        self.odom_pub.publish(odom_msg)

        # -- TF --
        if self.publish_tf:
            self._broadcast_tf(stamp)

        # -- 路径 --
        self._update_path(stamp)

    def _build_odom_msg(self, stamp):
        """构造 nav_msgs/Odometry 消息"""
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.odom_frame   # 父坐标系: odom
        odom.child_frame_id = self.base_frame     # 子坐标系: base

        # 位姿
        odom.pose.pose.position.x = self.cur_x
        odom.pose.pose.position.y = self.cur_y
        odom.pose.pose.position.z = self.cur_z

        q = quaternion_from_euler(self.cur_roll, self.cur_pitch, self.cur_yaw)
        odom.pose.pose.orientation.x = q[0]
        odom.pose.pose.orientation.y = q[1]
        odom.pose.pose.orientation.z = q[2]
        odom.pose.pose.orientation.w = q[3]

        # 协方差 (粗糙估计，设为单位阵)
        # position covariance [x, y, z] — 对角线
        odom.pose.covariance[0] = 0.01   # x 方差 0.1²
        odom.pose.covariance[7] = 0.01   # y 方差
        odom.pose.covariance[14] = 0.01  # z 方差
        # orientation covariance [roll, pitch, yaw] — 对角线
        odom.pose.covariance[21] = 0.001742  # (2.39°)²
        odom.pose.covariance[28] = 0.001742
        odom.pose.covariance[35] = 0.001742

        # 速度 (本节点不估计速度，留空)
        # 可以用帧间位姿差 / 时间差来估计，后续 EKF 融合时再做

        return odom

    def _broadcast_tf(self, stamp):
        """广播 odom → base TF"""
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = self.odom_frame
        t.child_frame_id = self.base_frame

        t.transform.translation.x = self.cur_x
        t.transform.translation.y = self.cur_y
        t.transform.translation.z = self.cur_z

        q = quaternion_from_euler(self.cur_roll, self.cur_pitch, self.cur_yaw)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        self.tf_broadcaster.sendTransform(t)

    def _update_path(self, stamp):
        """更新轨迹路径"""
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = stamp
        pose_stamped.header.frame_id = self.odom_frame
        pose_stamped.pose.position.x = self.cur_x
        pose_stamped.pose.position.y = self.cur_y
        pose_stamped.pose.position.z = self.cur_z

        q = quaternion_from_euler(self.cur_roll, self.cur_pitch, self.cur_yaw)
        pose_stamped.pose.orientation.x = q[0]
        pose_stamped.pose.orientation.y = q[1]
        pose_stamped.pose.orientation.z = q[2]
        pose_stamped.pose.orientation.w = q[3]

        self.path.poses.append(pose_stamped)
        self.path.header.stamp = stamp
        self.path_pub.publish(self.path)

    # ================================================================
    # 点云解析: PointCloud2 → numpy
    # ================================================================
    def _pointcloud2_to_numpy(self, msg):
        """解析 PointCloud2 为 numpy 数组"""
        gen = read_points(msg, field_names=('x', 'y', 'z', 'intensity'),
                          skip_nans=True)
        points_list = list(gen)
        if len(points_list) == 0:
            return None, False

        has_intensity = 'intensity' in {f.name for f in msg.fields}
        return np.array(points_list, dtype=np.float32), has_intensity

    # ================================================================
    # numpy → PointCloud2
    # ================================================================
    def _numpy_to_pointcloud2(self, points, stamp, frame_id,
                               has_intensity=True):
        """将 numpy 数组转为 PointCloud2"""
        header = Header()
        header.stamp = stamp
        header.frame_id = frame_id

        if has_intensity and points.shape[1] >= 4:
            fields = [
                PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1),
                PointField('intensity', 12, PointField.FLOAT32, 1),
            ]
            cloud_data = [tuple(p) for p in points[:, :4]]
        else:
            fields = [
                PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1),
            ]
            cloud_data = [(p[0], p[1], p[2]) for p in points]

        return create_cloud(header, fields, cloud_data)

    # ================================================================
    # 数学工具
    # ================================================================
    def _rotation_matrix_to_euler(self, R):
        """3x3 旋转矩阵 → (roll, pitch, yaw)"""
        # sy = sqrt(R[0,0]² + R[1,0]²)
        sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            roll = math.atan2(R[2, 1], R[2, 2])
            pitch = math.atan2(-R[2, 0], sy)
            yaw = math.atan2(R[1, 0], R[0, 0])
        else:
            roll = math.atan2(-R[1, 2], R[1, 1])
            pitch = math.atan2(-R[2, 0], sy)
            yaw = 0.0
        return roll, pitch, yaw

    def _euler_to_rotation_matrix(self, roll, pitch, yaw):
        """(roll, pitch, yaw) → 3x3 旋转矩阵 (ZYX 内旋)"""
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)

        R = np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp,     cp * sr,                cp * cr]
        ], dtype=np.float64)
        return R

    def quaternion_to_rotation_matrix(self, qx, qy, qz, qw):
        """四元数 → 3x3 旋转矩阵"""
        return self._euler_to_rotation_matrix(
            *euler_from_quaternion([qx, qy, qz, qw]))


# ====================================================================
# 主函数
# ====================================================================
def main():
    try:
        node = LidarOdometryNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("[lidar_odometry_node] 节点正常退出")
    except KeyboardInterrupt:
        rospy.loginfo("[lidar_odometry_node] 收到 Ctrl+C, 退出")


if __name__ == '__main__':
    main()
