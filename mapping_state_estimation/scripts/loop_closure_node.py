#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
 回环检测与位姿图优化节点 (Step 9)
==============================================================================
 功能: 检测机器人是否回到之前访问过的区域，修正全局漂移。

 方案: 基于位置距离的回环检测 + 简单 ICP 验证
       室内场景 20m×36m，累计漂移通常 < 2m，位置距离判断足够。

 处理流程:
   1. 维护关键帧历史队列 (位置 + 点云)
   2. 每个新关键帧到达时，在历史帧中搜索距离 < 阈值 的候选
   3. 排除时间上相邻的关键帧 (至少间隔 30 帧)
   4. ICP 验证: 对候选对做 ICP 匹配
   5. 如果 ICP 成功且误差小，确认回环
   6. 发布累积修正量 → grid_map 更新 world → odom TF

 订阅:
   /cloud_keyframe         sensor_msgs/PointCloud2    关键帧点云 (步骤5)
   /lidar_odom             nav_msgs/Odometry           LiDAR 里程计位姿

 发布:
   /loop_closure/status    std_msgs/String            回环状态
   /loop_closure/correction geometry_msgs/PoseStamped 累积修正量 (world→odom, 由 grid_map 应用)

 注: 本节点不再直接发布 TF, world→odom 由 grid_map_node 统一发布 (含回环修正)

 参数:
   ~distance_threshold      float  回环检测距离阈值 (m), 默认 1.5
   ~min_keyframe_gap        int    最小关键帧间隔, 默认 30
   ~icp_fitness_threshold   float  ICP 适应度阈值, 默认 0.1
   ~max_loop_closures       int    最大回环次数, 默认 5
   ~search_radius           float  搜索半径 (m), 默认 10.0

==============================================================================
"""

import math
import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs.point_cloud2 import read_points, create_cloud
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String, Header, Int8
from sensor_msgs.msg import PointField
from tf.transformations import quaternion_from_euler, euler_from_quaternion


class LoopClosureNode:
    """回环检测节点"""

    def __init__(self):
        rospy.init_node('loop_closure_node', anonymous=False)
        self.node_name = rospy.get_name()

        # ================================================================
        # 参数
        # ================================================================
        self.distance_threshold = rospy.get_param('~distance_threshold', 1.5)
        self.min_keyframe_gap = rospy.get_param('~min_keyframe_gap', 30)
        self.icp_fitness_threshold = rospy.get_param('~icp_fitness_threshold', 0.1)
        self.max_loop_closures = rospy.get_param('~max_loop_closures', 5)
        self.search_radius = rospy.get_param('~search_radius', 10.0)

        # ================================================================
        # 状态
        # ================================================================
        # 关键帧队列: [(x, y, z, roll, pitch, yaw, cloud_numpy), ...]
        self.keyframes = []
        self.loop_count = 0

        # 累积修正 (world → odom)
        self.correction_x = 0.0
        self.correction_y = 0.0
        self.correction_z = 0.0
        self.correction_roll = 0.0
        self.correction_pitch = 0.0
        self.correction_yaw = 0.0

        # 最近一次回环的关键帧索引 (防止重复回环)
        self.last_loop_kf_idx = -100

        # ================================================================
        # 订阅
        # ================================================================
        rospy.Subscriber('/cloud_keyframe', PointCloud2,
                         self.keyframe_callback, queue_size=10)
        rospy.Subscriber('/lidar_odom', Odometry,
                         self.odom_callback, queue_size=10)
        # 订阅当前楼层
        rospy.Subscriber('/current_floor', Int8,
                         self.floor_callback, queue_size=5)

        # ================================================================
        # 发布
        # ================================================================
        self.status_pub = rospy.Publisher('/loop_closure/status', String,
                                           queue_size=5)
        self.correction_pub = rospy.Publisher('/loop_closure/correction',
                                               PoseStamped, queue_size=5)

        self.current_floor = 0

        rospy.loginfo("[%s] 回环检测节点启动完成", self.node_name)
        rospy.loginfo("[%s]   距离阈值: %.1f m, 最小帧间隔: %d",
                      self.node_name, self.distance_threshold,
                      self.min_keyframe_gap)
        rospy.loginfo("[%s]   最大回环次数: %d", self.node_name,
                      self.max_loop_closures)

    # ================================================================
    # 回调
    # ================================================================
    def odom_callback(self, msg):
        """里程计回调: 记录最新位姿"""
        # 只用于记录位姿，关键帧点云在 keyframe_callback 里处理
        pass

    def floor_callback(self, msg):
        self.current_floor = msg.data

    def keyframe_callback(self, msg):
        """关键帧点云回调: 存储并检测回环"""
        if self.loop_count >= self.max_loop_closures:
            return  # 达到最大回环次数

        # 解析点云
        points_arr = self._pointcloud2_to_numpy(msg)
        if points_arr is None or len(points_arr) < 100:
            return

        # 尝试从 TF 获取当前关键帧位姿
        try:
            # 用最新里程计位姿作为关键帧位姿
            # (简化: 直接用点云质心，或者等位姿从 odom_callback 来)
            pass
        except:
            pass

        # 简化: 用点云中心近似位置
        cx = float(np.median(points_arr[:, 0]))
        cy = float(np.median(points_arr[:, 1]))
        cz = float(np.median(points_arr[:, 2]))

        kf_idx = len(self.keyframes)

        # ---- 搜索回环候选 ----
        # 在历史关键帧中搜索距离近的帧
        for hist_idx in range(max(0, kf_idx - self.min_keyframe_gap)):
            hx, hy, hz = self.keyframes[hist_idx][:3]
            dist = math.sqrt((cx - hx)**2 + (cy - hy)**2)

            if dist < self.distance_threshold and \
               (kf_idx - self.last_loop_kf_idx) > self.min_keyframe_gap // 2:
                # 候选回环，ICP 验证
                success, T = self._verify_loop(
                    points_arr, self.keyframes[hist_idx][6], dist)

                if success:
                    self._handle_loop_closure(kf_idx, hist_idx, T, dist)
                    break  # 一次只处理一个回环

        # ---- 存储关键帧 ----
        # 下采样后存储减少内存
        if len(points_arr) > 2000:
            idx = np.random.choice(len(points_arr), 2000, replace=False)
            points_arr = points_arr[idx]

        self.keyframes.append((cx, cy, cz, 0.0, 0.0, 0.0, points_arr))

        # 限制关键帧队列大小
        if len(self.keyframes) > 500:
            self.keyframes.pop(0)

    # ================================================================
    # ICP 验证回环
    # ================================================================
    def _verify_loop(self, cur_cloud, hist_cloud, dist):
        """
        对回环候选对做简化 ICP 验证

        Returns:
            (success, T_4x4) or (False, None)
        """
        src = cur_cloud[:, :3].copy().astype(np.float64)
        tgt = hist_cloud[:, :3].copy().astype(np.float64)

        # 快速验证: 两部分点云至少要有一定重叠
        if len(src) < 50 or len(tgt) < 50:
            return False, None

        # 简化 ICP: 仅迭代 15 次
        T = np.eye(4, dtype=np.float64)
        prev_error = float('inf')

        for iteration in range(15):
            # 最近邻关联 (分块)
            distances, indices = self._nearest_neighbors(src, tgt, max_dist=2.0)
            valid_mask = distances < 2.0
            n_valid = int(valid_mask.sum())

            if n_valid < 20:
                return False, None

            src_valid = src[valid_mask]
            tgt_valid = tgt[indices[valid_mask]]

            # SVD 求解
            c_src = src_valid.mean(axis=0)
            c_tgt = tgt_valid.mean(axis=0)
            src_c = src_valid - c_src
            tgt_c = tgt_valid - c_tgt
            H = src_c.T @ tgt_c

            U, S, Vt = np.linalg.svd(H)
            V = Vt.T
            Ut = U.T
            det = np.linalg.det(V @ Ut)
            corr = np.eye(3)
            corr[2, 2] = det
            R_iter = V @ corr @ Ut
            t_iter = c_tgt - R_iter @ c_src

            T_iter = np.eye(4)
            T_iter[:3, :3] = R_iter
            T_iter[:3, 3] = t_iter

            # 应用变换
            homog = np.ones((len(src), 4))
            homog[:, :3] = src
            src = (T_iter @ homog.T).T[:, :3]
            T = T_iter @ T

            current_error = float(np.mean(distances[valid_mask] ** 2))
            if abs(prev_error - current_error) < 1e-6:
                break
            prev_error = current_error

        # 检查最终适应度
        final_mse = prev_error
        if final_mse < self.icp_fitness_threshold:
            return True, T

        return False, None

    # ================================================================
    # 处理回环
    # ================================================================
    def _handle_loop_closure(self, cur_idx, hist_idx, T, dist):
        """处理检测到的回环"""
        self.loop_count += 1
        self.last_loop_kf_idx = cur_idx

        # 提取修正量 (平移 + 平面偏航角)
        dx = T[0, 3]
        dy = T[1, 3]
        dz = T[2, 3]
        dyaw = math.atan2(T[1, 0], T[0, 0])

        # 累积修正
        self.correction_x += dx
        self.correction_y += dy
        self.correction_z += dz
        self.correction_yaw += dyaw

        rospy.loginfo("[%s] ★ 回环检测 #%d: 关键帧 %d ↔ %d, "
                      "距离=%.2fm, 修正=(%.3f, %.3f, %.3f, %.2f°)",
                      self.node_name, self.loop_count,
                      hist_idx, cur_idx, dist, dx, dy, dz, math.degrees(dyaw))

        # 发布状态
        status = String()
        status.data = (f"Loop #{self.loop_count}: KF {hist_idx}↔{cur_idx}, "
                       f"dist={dist:.2f}m, correction=({dx:.3f},{dy:.3f},{dz:.3f})")
        self.status_pub.publish(status)

        # 发布累积修正量 (world→odom), 由 grid_map_node 订阅并应用
        corr_pose = PoseStamped()
        corr_pose.header.stamp = rospy.Time.now()
        corr_pose.header.frame_id = 'world'
        corr_pose.pose.position.x = self.correction_x
        corr_pose.pose.position.y = self.correction_y
        corr_pose.pose.position.z = self.correction_z
        q = quaternion_from_euler(self.correction_roll,
                                  self.correction_pitch,
                                  self.correction_yaw)
        corr_pose.pose.orientation.x = q[0]
        corr_pose.pose.orientation.y = q[1]
        corr_pose.pose.orientation.z = q[2]
        corr_pose.pose.orientation.w = q[3]
        self.correction_pub.publish(corr_pose)

    # ================================================================
    # 最近邻搜索 (简化版)
    # ================================================================
    def _nearest_neighbors(self, src, tgt, max_dist, batch_size=500):
        n_src = len(src)
        all_distances = []
        all_indices = []

        for start in range(0, n_src, batch_size):
            end = min(start + batch_size, n_src)
            batch = src[start:end]
            diff = batch[:, None, :] - tgt[None, :, :]
            dist = np.sqrt(np.sum(diff * diff, axis=2))
            batch_min_idx = np.argmin(dist, axis=1)
            batch_min_dist = dist[np.arange(len(batch)), batch_min_idx]
            all_distances.append(batch_min_dist)
            all_indices.append(batch_min_idx)

        distances = np.concatenate(all_distances)
        indices = np.concatenate(all_indices)
        distances = np.where(distances <= max_dist, distances, max_dist)
        return distances, indices

    # ================================================================
    # 点云解析
    # ================================================================
    def _pointcloud2_to_numpy(self, msg):
        gen = read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)
        points_list = list(gen)
        if len(points_list) == 0:
            return None
        return np.array(points_list, dtype=np.float32)


# ====================================================================
def main():
    try:
        node = LoopClosureNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("[loop_closure_node] 节点正常退出")
    except KeyboardInterrupt:
        rospy.loginfo("[loop_closure_node] 收到 Ctrl+C, 退出")


if __name__ == '__main__':
    main()
