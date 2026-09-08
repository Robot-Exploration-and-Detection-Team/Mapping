#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
 点云预处理节点 (Step 4)
==============================================================================
 功能: 接收 Livox 原始点云，依次完成：
       1. 坐标变换: laser_livox → base (通过 TF 查询外参)
       2. 运动去畸变: 利用 IMU 角速度补偿扫描期间的机器人运动
       3. 离群点去除: 基于体素的统计滤波 (PCL-free, 纯 numpy 实现)
       4. 发布处理后的点云 /cloud_base

 订阅:
   /livox/Pointcloud2     sensor_msgs/PointCloud2    Livox 原始点云 (10 Hz)
   /trunk_imu             sensor_msgs/Imu            机体 IMU (1000 Hz, 用于去畸变)

 发布:
   /cloud_base            sensor_msgs/PointCloud2    base 坐标系下的干净点云
   /cloud_base_debug      sensor_msgs/PointCloud2    含调试信息 (每个点的 label 字段标记是否离群点)

 参数:
   ~lidar_frame           str   雷达坐标系, 默认 "laser_livox"
   ~target_frame          str   目标坐标系, 默认 "base"
   ~max_range             float 最大有效距离 (m), 默认 40.0
   ~min_range             float 最小有效距离 (m), 默认 0.1
   ~deskew_enabled        bool  是否启用去畸变, 默认 true
   ~outlier_radius        float 离群点检测半径 (m), 默认 0.1
   ~outlier_min_neighbors int   邻域最小点数, 默认 3
   ~voxel_leaf_size       float 体素下采样分辨率 (m), 默认 0.05

==============================================================================
"""

import math
import numpy as np
import rospy
import tf2_ros
import tf2_geometry_msgs
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs.point_cloud2 import read_points, create_cloud
from sensor_msgs.msg import Imu
from std_msgs.msg import Header

try:
    from tf.transformations import euler_from_quaternion, quaternion_matrix
except ImportError:
    from tf.transformations import euler_from_quaternion, quaternion_matrix


class CloudPreprocessNode:
    """点云预处理节点"""

    def __init__(self):
        rospy.init_node('cloud_preprocess_node', anonymous=False)
        self.node_name = rospy.get_name()

        # ============================================================
        # 参数
        # ============================================================
        self.lidar_frame = rospy.get_param('~lidar_frame', 'laser_livox')
        self.target_frame = rospy.get_param('~target_frame', 'base')
        self.max_range = rospy.get_param('~max_range', 40.0)
        self.min_range = rospy.get_param('~min_range', 0.1)
        self.deskew_enabled = rospy.get_param('~deskew_enabled', True)
        self.outlier_radius = rospy.get_param('~outlier_radius', 0.1)
        self.outlier_min_neighbors = rospy.get_param('~outlier_min_neighbors', 3)
        self.voxel_leaf_size = rospy.get_param('~voxel_leaf_size', 0.02)

        rospy.loginfo("[%s] 参数:", self.node_name)
        rospy.loginfo("[%s]   坐标系: %s → %s", self.node_name,
                      self.lidar_frame, self.target_frame)
        rospy.loginfo("[%s]   距离范围: %.1f ~ %.1f m", self.node_name,
                      self.min_range, self.max_range)
        rospy.loginfo("[%s]   去畸变: %s", self.node_name, self.deskew_enabled)
        rospy.loginfo("[%s]   离群点检测: 半径=%.2fm, 最小邻居=%d",
                      self.node_name, self.outlier_radius, self.outlier_min_neighbors)

        # ============================================================
        # TF
        # ============================================================
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(5.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # ============================================================
        # IMU 角速度缓冲区 (用于去畸变)
        # 存储格式: [(timestamp_sec, wx, wy, wz), ...]
        # ============================================================
        self.imu_buffer = []
        self.imu_buffer_max_age = 1.0  # 只保留最近 1 秒的 IMU 数据

        # ============================================================
        # 订阅
        # ============================================================
        rospy.Subscriber('/livox/Pointcloud2', PointCloud2,
                         self.cloud_callback, queue_size=10)
        rospy.Subscriber('/trunk_imu', Imu,
                         self.imu_callback, queue_size=100)

        # ============================================================
        # 发布
        # ============================================================
        self.cloud_pub = rospy.Publisher('/cloud_base', PointCloud2, queue_size=10)
        self.cloud_debug_pub = rospy.Publisher(
            '/cloud_base_debug', PointCloud2, queue_size=10)

        # 固定外参缓存 (laser_livox → base, 静态变换, 查一次即可)
        self.static_transform = None  # (translation, rotation_matrix)

        rospy.loginfo("[%s] 点云预处理节点启动完成", self.node_name)
        rospy.loginfo("[%s] 订阅: /livox/Pointcloud2, /trunk_imu", self.node_name)
        rospy.loginfo("[%s] 发布: /cloud_base", self.node_name)

    # ================================================================
    # IMU 回调
    # ================================================================
    def imu_callback(self, msg):
        """
        存储 IMU 角速度到缓冲区，用于去畸变时查询
        维护一个时间窗口内的滑动缓冲区
        """
        t = msg.header.stamp.to_sec()
        wx = msg.angular_velocity.x
        wy = msg.angular_velocity.y
        wz = msg.angular_velocity.z
        self.imu_buffer.append((t, wx, wy, wz))

        # 清理过期数据
        cutoff = t - self.imu_buffer_max_age
        self.imu_buffer = [x for x in self.imu_buffer if x[0] > cutoff]

    # ================================================================
    # 点云回调 (主处理流水线)
    # ================================================================
    def cloud_callback(self, msg):
        """
        主处理流水线:
          raw cloud → 坐标变换 → 去畸变 → 离群点去除 → publish
        """
        # ---- 0. 解析点云为 numpy 数组 ----
        points_arr, has_intensity = self._pointcloud2_to_numpy(msg)
        if points_arr is None or len(points_arr) == 0:
            rospy.logwarn_throttle(5.0, "[%s] 收到空点云", self.node_name)
            return

        n_raw = len(points_arr)
        cloud_t = msg.header.stamp.to_sec()

        # ---- 1. 距离过滤 (NaN / 超出量程) ----
        points_arr = self._range_filter(points_arr)
        n_filtered = len(points_arr)

        # ---- 2. 坐标变换: msg.header.frame_id → base ----
        points_arr = self._transform_to_base(points_arr, msg.header.stamp,
                                              msg.header.frame_id)
        if points_arr is None:
            return  # TF 查询失败

        # ---- 3. 运动去畸变 ----
        if self.deskew_enabled:
            points_arr = self._deskew(points_arr, cloud_t)

        # ---- 4. 体素下采样 ----
        points_arr = self._voxel_downsample(points_arr)

        # ---- 5. 离群点去除 ----
        inlier_mask = self._radius_outlier_filter(points_arr)
        n_inliers = int(inlier_mask.sum())
        n_outliers = len(points_arr) - n_inliers

        # ---- 6. 发布 /cloud_base ----
        inlier_points = points_arr[inlier_mask]
        cloud_msg = self._numpy_to_pointcloud2(inlier_points, msg.header, has_intensity)
        self.cloud_pub.publish(cloud_msg)

        # ---- 7. 调试发布 (标记离群点) ----
        if self.cloud_debug_pub.get_num_connections() > 0:
            debug_cloud = self._make_debug_cloud(points_arr, inlier_mask, msg.header)
            self.cloud_debug_pub.publish(debug_cloud)

        # 统计
        rospy.loginfo_throttle(10.0,
                               "[%s] 原始:%d → 距离过滤:%d → 下采样+去离群:%d (剔除:%d)",
                               self.node_name, n_raw, n_filtered,
                               n_inliers, n_outliers)

    # ================================================================
    # 1. 点云解析: PointCloud2 → numpy
    # ================================================================
    def _pointcloud2_to_numpy(self, msg):
        """
        将 sensor_msgs/PointCloud2 转为 numpy 数组
        Livox PointCloud2 格式: x, y, z, intensity (float32 × 4)

        Returns:
            numpy.ndarray shape (N, 3+): [x, y, z, (intensity)]
            has_intensity: bool
        """
        # 读取原始数据
        gen = read_points(msg, field_names=('x', 'y', 'z', 'intensity'),
                          skip_nans=False)
        points_list = list(gen)

        if len(points_list) == 0:
            return None, False

        # 检查是否有 intensity
        has_intensity = 'intensity' in {f.name for f in msg.fields}

        return np.array(points_list, dtype=np.float32), has_intensity

    # ================================================================
    # 2. 距离过滤
    # ================================================================
    def _range_filter(self, points):
        """
        移除 NaN 点和超出量程的点
        """
        # 去 NaN
        nan_mask = np.isnan(points).any(axis=1)
        points = points[~nan_mask]

        # 计算距离
        dist = np.linalg.norm(points[:, :3], axis=1)

        # 距离过滤
        valid = (dist >= self.min_range) & (dist <= self.max_range)
        return points[valid]

    # ================================================================
    # 3. 坐标变换: laser_livox → base
    # ================================================================
    def _transform_to_base(self, points, stamp, source_frame=None):
        """
        将点云从 source_frame 变换到 base 坐标系

        优先使用传入的 source_frame (点云消息头中的 frame_id)，
        若为 None 则回退到 self.lidar_frame 参数。

        如果 source_frame 已经是 base 则直接返回 (零变换)。
        """
        if source_frame is None:
            source_frame = self.lidar_frame

        # 已在 base 系: 跳过变换
        if source_frame == self.target_frame:
            return points

        # 查询 TF: target_frame ← source_frame
        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,      # target: base
                source_frame,           # source: 点云实际坐标系
                rospy.Time(0),
                rospy.Duration(3.0))
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(5.0,
                                   "[%s] TF 查询失败 (%s→%s): %s",
                                   self.node_name,
                                   source_frame, self.target_frame, e)
            return None

        t = transform.transform.translation
        q = transform.transform.rotation
        translation = np.array([t.x, t.y, t.z], dtype=np.float32)

        qx, qy, qz, qw = q.x, q.y, q.z, q.w
        R = quaternion_matrix([qx, qy, qz, qw])[:3, :3].astype(np.float32)

        transformed = (R @ points[:, :3].T).T + translation
        points[:, :3] = transformed

        return points

    # ================================================================
    # 4. 运动去畸变 (Deskew) — 向量化优化版
    # ================================================================
    def _deskew(self, points, cloud_t):
        """
        利用 IMU 角速度补偿扫描期间的机器人运动 (向量化实现)

        原理:
          假设扫描持续 ~100ms，所有点均匀分布在时间窗口内。
          用 IMU 平均角速度计算每个点的反向旋转矩阵。
          向量化替代逐点 Python 循环，大幅降低 CPU 开销。
        """
        n = len(points)
        if n < 2:
            return points

        scan_duration = 0.1
        offsets = np.linspace(0, scan_duration, n, dtype=np.float64)
        delta_t = scan_duration - offsets  # (N,) 到帧末的时间差

        avg_wx, avg_wy, avg_wz = self._get_average_angular_velocity(cloud_t)

        if avg_wx == 0.0 and avg_wy == 0.0 and avg_wz == 0.0:
            return points

        # 批量构造小角度旋转矩阵
        # 对每个点: angle = w * delta_t[i]
        rx = -avg_wx * delta_t
        ry = -avg_wy * delta_t
        rz = -avg_wz * delta_t

        # 向量化: 小角度近似旋转矩阵 R ≈ I + skew(angle)
        # p_new = R @ p = p + skew(angle) @ p
        # skew = [[0, -rz, ry], [rz, 0, -rx], [-ry, rx, 0]]
        # R @ p = p + cross(angle, p)  其中 angle = [rx, ry, rz]^T 但符号需注意

        # 实际: R_deskew @ p = p + angle × p = p + cross([rx, ry, rz], p)
        # 因为 skew-symmetric 矩阵乘法等价于叉积
        # cross([rx, ry, rz], p) = [ry*pz - rz*py, rz*px - rx*pz, rx*py - ry*px]

        px = points[:, 0]
        py = points[:, 1]
        pz = points[:, 2]

        points[:, 0] = px + (ry * pz - rz * py)
        points[:, 1] = py + (rz * px - rx * pz)
        points[:, 2] = pz + (rx * py - ry * px)

        return points

    def _get_average_angular_velocity(self, cloud_t):
        """
        获取扫描窗口内的平均 IMU 角速度
        cloud_t: 扫描结束时间 (s)
        scan_duration: 0.1s, 查询范围为 [cloud_t - 0.1, cloud_t]
        """
        window_start = cloud_t - 0.15  # 稍宽一点确保覆盖
        window_end = cloud_t + 0.02

        # 筛选时间窗口内的 IMU 数据
        window_data = [(wx, wy, wz) for t, wx, wy, wz in self.imu_buffer
                       if window_start <= t <= window_end]

        if len(window_data) < 5:
            # 数据不足, 尝试放宽窗口到 0.5s
            window_data = [(wx, wy, wz) for t, wx, wy, wz in self.imu_buffer
                           if abs(t - cloud_t) < 0.5]
        if len(window_data) < 5:
            return 0.0, 0.0, 0.0

        avg = np.mean(window_data, axis=0)
        return float(avg[0]), float(avg[1]), float(avg[2])

    # ================================================================
    # 5. 体素下采样
    # ================================================================
    def _voxel_downsample(self, points):
        """
        基于体素网格的下采样，减少点云密度

        方法: 将点索引到体素网格，每个体素只取中心最近的 1 个点
        """
        if len(points) < 100:
            return points

        v = self.voxel_leaf_size
        if v <= 0:
            return points

        # 计算每个点所在的体素索引
        voxel_idx = np.floor(points[:, :3] / v).astype(np.int32)

        # 用字典去重，每个体素保留第一个点（或体素中心点）
        # 这里取每个体素内点的均值中心，质量更好
        voxel_dict = {}
        for i in range(len(points)):
            key = tuple(voxel_idx[i])
            if key not in voxel_dict:
                voxel_dict[key] = [points[i]]
            else:
                voxel_dict[key].append(points[i])

        # 每个体素取中心点（平均值）
        downsampled = []
        for pts in voxel_dict.values():
            if len(pts) == 1:
                downsampled.append(pts[0])
            else:
                downsampled.append(np.mean(pts, axis=0))

        rospy.logdebug("[%s] 下采样: %d → %d (体素 %.2fm)",
                       self.node_name, len(points), len(downsampled), v)

        return np.array(downsampled, dtype=np.float32)

    # ================================================================
    # 6. 离群点去除 (Grid-Accelerated Radius Outlier Removal)
    # ================================================================
    def _radius_outlier_filter(self, points):
        """
        基于体素网格加速的离群点检测 (O(N·K) 替代 O(N²))

        方法:
          1. 将点分配到空间网格
          2. 对每个点, 只检查其所在网格及相邻 26 个网格内的邻居
          3. 邻居数 < threshold → 离群点

        网格大小 = outlier_radius (确保半径内的邻居一定在同一或相邻网格中)
        """
        n = len(points)
        if n == 0:
            return np.zeros(0, dtype=bool)

        xyz = points[:, :3]
        cell_size = self.outlier_radius
        radius_sq = self.outlier_radius ** 2

        # 构建空间网格字典: key=(vx, vy, vz) → list of point indices
        grid = {}
        for i in range(n):
            vx = int(xyz[i, 0] / cell_size)
            vy = int(xyz[i, 1] / cell_size)
            vz = int(xyz[i, 2] / cell_size)
            key = (vx, vy, vz)
            if key not in grid:
                grid[key] = []
            grid[key].append(i)

        # 对每个点检查邻居
        inlier = np.zeros(n, dtype=bool)
        # 相邻网格偏移 (27 个: 自身 + 26 个相邻)
        offsets = [(dx, dy, dz) for dx in (-1, 0, 1)
                              for dy in (-1, 0, 1)
                              for dz in (-1, 0, 1)]

        for i in range(n):
            vx = int(xyz[i, 0] / cell_size)
            vy = int(xyz[i, 1] / cell_size)
            vz = int(xyz[i, 2] / cell_size)
            count = 0

            for dx, dy, dz in offsets:
                key = (vx + dx, vy + dy, vz + dz)
                if key in grid:
                    for j in grid[key]:
                        dist_sq = ((xyz[i, 0] - xyz[j, 0]) ** 2 +
                                   (xyz[i, 1] - xyz[j, 1]) ** 2 +
                                   (xyz[i, 2] - xyz[j, 2]) ** 2)
                        if dist_sq <= radius_sq:
                            count += 1
                            if count >= self.outlier_min_neighbors:
                                inlier[i] = True
                                break
                if inlier[i]:
                    break

        return inlier

    # ================================================================
    # 7. 发布: numpy → PointCloud2
    # ================================================================
    def _numpy_to_pointcloud2(self, points, original_header, has_intensity=True):
        """
        将 numpy 数组转换回 sensor_msgs/PointCloud2
        """
        header = Header()
        header.stamp = original_header.stamp
        header.frame_id = self.target_frame  # 目标坐标系

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

    def _make_debug_cloud(self, points, inlier_mask, original_header):
        """
        生成调试点云: 离群点标为红色 (high intensity), 内点标为蓝色 (low intensity)
        """
        header = Header()
        header.stamp = original_header.stamp
        header.frame_id = self.target_frame

        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('intensity', 12, PointField.FLOAT32, 1),
        ]

        cloud_data = []
        for i in range(len(points)):
            x, y, z = points[i, 0], points[i, 1], points[i, 2]
            # 内点 intensity=1 (蓝色), 离群点 intensity=100 (红色)
            ii = 1.0 if inlier_mask[i] else 100.0
            cloud_data.append((x, y, z, ii))

        return create_cloud(header, fields, cloud_data)


# ====================================================================
def main():
    try:
        node = CloudPreprocessNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("[cloud_preprocess_node] 节点正常退出")
    except KeyboardInterrupt:
        rospy.loginfo("[cloud_preprocess_node] 收到 Ctrl+C, 退出")


if __name__ == '__main__':
    main()
