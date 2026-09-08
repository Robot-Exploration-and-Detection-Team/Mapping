#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
 IMU 姿态解算节点  (Step 3)
==============================================================================
 功能: 订阅 /trunk_imu，从机体 IMU 数据实时解算机器人的 roll/pitch/yaw
       支持两种模式:
         模式A (基础): 直接从 orientation 四元数转换 → 欧拉角
         模式B (进阶): 互补滤波 — 角速度积分短期预测 + orientation 长期修正

 订阅话题:
   /trunk_imu          sensor_msgs/Imu       机体 IMU (1000 Hz)
   /ground_truth/base_w (可选)                真值对比，仅调试

 发布话题:
   /imu/attitude       geometry_msgs/Vector3Stamped   实时姿态 (roll, pitch, yaw)
   /imu/attitude_filtered   geometry_msgs/Vector3Stamped  滤波后姿态 (仅模式B)

 参数 (ROS param):
   ~mode               str  "basic" (默认) 或 "complementary"
   ~alpha              float 互补滤波系数, 范围 [0, 1], 默认 0.98
                             越大越信任 orientation，越小越信任角速度积分
   ~print_hz           float 终端打印频率 Hz, 默认 10.0
   ~use_ground_truth   bool  是否对比真值, 默认 false

==============================================================================
 坐标系约定 (ROS REP 103):
   X 向前 — roll  (绕 X 轴旋转, 即绕前进方向的翻滚)
   Y 向左 — pitch (绕 Y 轴旋转, 即绕左方向的俯仰)
   Z 向上 — yaw   (绕 Z 轴旋转, 即绕上方向的偏航)

 注意: 所有角度单位为 弧度 (rad), 打印时转换为 度 (°)
==============================================================================
"""

import math
import rospy
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3Stamped
import tf2_ros
from tf.transformations import euler_from_quaternion, quaternion_from_euler


class IMUAttitudeNode:
    """IMU 姿态解算节点

    从 /trunk_imu 读取四元数姿态和角速度，
    解算出机器人 roll/pitch/yaw 并实时发布。
    """

    def __init__(self):
        # ============================================================
        # 1. ROS 节点初始化
        # ============================================================
        rospy.init_node('imu_attitude_node', anonymous=False)
        self.node_name = rospy.get_name()

        # ============================================================
        # 2. 读取参数
        # ============================================================
        # 模式: "basic" 直接四元数转换 / "complementary" 互补滤波
        self.mode = rospy.get_param('~mode', 'basic')
        # 互补滤波系数: 融合权重 alpha * orientation + (1-alpha) * gyro_integral
        self.alpha = rospy.get_param('~alpha', 0.98)
        # 终端打印频率, 默认 10 Hz (IMU 数据频率 1000 Hz, 全部打印会刷屏)
        self.print_hz = rospy.get_param('~print_hz', 10.0)
        # 是否与 ground_truth 对比 (仅调试阶段使用, 比赛禁止)
        self.use_ground_truth = rospy.get_param('~use_ground_truth', False)

        rospy.loginfo("[%s] 模式: %s", self.node_name, self.mode)
        rospy.loginfo("[%s] 互补滤波系数 alpha: %.3f", self.node_name, self.alpha)
        rospy.loginfo("[%s] 打印频率: %.1f Hz", self.node_name, self.print_hz)

        # ============================================================
        # 3. 状态变量
        # ============================================================
        # 当前姿态 (roll, pitch, yaw) 单位: 弧度
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

        # 滤波后姿态 (仅 complementary 模式使用)
        self.roll_f = 0.0
        self.pitch_f = 0.0
        self.yaw_f = 0.0

        # 上一帧时间戳, 用于角速度积分的时间差计算
        self.last_time = None

        # 互补滤波是否已初始化 (第一次收到 imu 消息时用 orientation 初始化)
        self.filter_initialized = False

        # 地面真值姿态 (调试对比用)
        self.gt_roll = 0.0
        self.gt_pitch = 0.0
        self.gt_yaw = 0.0

        # 打印计数器控制
        self.last_print_time = rospy.Time.now()

        # ============================================================
        # 4. 订阅者
        # ============================================================
        # 主 IMU — 机体躯干 IMU, 1000 Hz
        rospy.Subscriber('/trunk_imu', Imu, self.imu_callback, queue_size=100)

        # 可选: 地面真值 (调试对比用, 比赛禁止)
        if self.use_ground_truth:
            rospy.Subscriber('/ground_truth/base_w', Imu, self.ground_truth_callback,
                             queue_size=10)
            rospy.loginfo("[%s] 已订阅 /ground_truth/base_w 用于调试对比", self.node_name)

        # ============================================================
        # 5. 发布者
        # ============================================================
        # 原始姿态 (basic 模式 = 直接四元数转换; complementary 模式 = 滤波后姿态)
        self.attitude_pub = rospy.Publisher('/imu/attitude', Vector3Stamped, queue_size=10)

        # 滤波后姿态 (仅 complementary 模式发布, 方便对比原始和滤波效果)
        if self.mode == 'complementary':
            self.attitude_filtered_pub = rospy.Publisher(
                '/imu/attitude_filtered', Vector3Stamped, queue_size=10)

        # ============================================================
        # 6. TF2 Buffer (用于监听坐标系变换, 后续步骤会用到)
        # ============================================================
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        rospy.loginfo("[%s] IMU 姿态解算节点启动完成", self.node_name)
        rospy.loginfo("[%s] 订阅: /trunk_imu", self.node_name)
        rospy.loginfo("[%s] 发布: %s", self.node_name, '/imu/attitude')
        if self.mode == 'complementary':
            rospy.loginfo("[%s] 发布: %s", self.node_name, '/imu/attitude_filtered')

    # ================================================================
    # 回调函数
    # ================================================================

    def imu_callback(self, msg):
        """
        /trunk_imu 回调 (1000 Hz)

        每收到一帧 IMU 数据就调用一次。
        参数 msg 是 sensor_msgs/Imu 类型, 包含:
          - orientation:   geometry_msgs/Quaternion  姿态四元数
          - angular_velocity:  geometry_msgs/Vector3     角速度 (rad/s)
          - linear_acceleration: geometry_msgs/Vector3   线加速度 (m/s²)
          - orientation_covariance:    float32[9]       姿态协方差矩阵
          - angular_velocity_covariance: float32[9]     角速度协方差矩阵
          - linear_acceleration_covariance: float32[9]  线加速度协方差矩阵
        """
        # ----- 提取四元数 -----
        q = msg.orientation
        qx, qy, qz, qw = q.x, q.y, q.z, q.w

        # ----- 四元数 → 欧拉角 (roll, pitch, yaw) -----
        # tf.transformations.euler_from_quaternion 返回 (roll, pitch, yaw)
        # 旋转顺序: Rz(yaw) * Ry(pitch) * Rx(roll) — 即 ZYX 内旋
        self.roll, self.pitch, self.yaw = euler_from_quaternion([qx, qy, qz, qw])

        # ----- 根据模式处理 -----
        if self.mode == 'complementary':
            self._complementary_filter(msg)
        else:
            # basic 模式: 直接使用四元数转换结果, 无需额外处理
            pass

        # ----- 发布姿态 -----
        self._publish_attitude(msg)

        # ----- 定时打印到终端 -----
        now = rospy.Time.now()
        if (now - self.last_print_time).to_sec() >= 1.0 / self.print_hz:
            self._print_status()
            self.last_print_time = now

        # ----- 更新时间戳 -----
        self.last_time = msg.header.stamp

    def ground_truth_callback(self, msg):
        """地面真值回调 (仅调试)

        从裁判通道获取机器人真实位姿。
        比赛时禁止订阅此话题。
        """
        q = msg.orientation
        self.gt_roll, self.gt_pitch, self.gt_yaw = euler_from_quaternion(
            [q.x, q.y, q.z, q.w])

    # ================================================================
    # 互补滤波
    # ================================================================

    def _complementary_filter(self, msg):
        """
        互补滤波器: 融合 orientation (低频准) 与角速度积分 (高频快)

        原理:
          姿态估计 = α × orientation_measured + (1-α) × gyro_integrated

          其中:
          - orientation_measured: 直接来自 IMU 消息的四元数 → 欧拉角
            Gazebo 仿真中这个值很准，低频无漂移，但有高频噪声
          - gyro_integrated:      对 gyro 角速度做数值积分得到的姿态增量
            高频响应快，但长时间积分会漂移
          - α (alpha):           融合系数 [0, 1]
            α → 1: 更信任 orientation (仿真环境下推荐较大值)
            α → 0: 更信任 gyro 积分 (真实IMU噪声大时)

        互补滤波的物理直觉:
          陀螺仪短期准、长期漂; 加速度计/磁力计长期准、短期噪
          高速滤波器 + 低通滤波器 → 互补 → 全频带覆盖

        首次调用时用 orientation 初始化滤波状态。
        """
        # 提取角速度 (单位: rad/s)
        gx = msg.angular_velocity.x  # 绕 X 轴的角速度 → roll 变化率
        gy = msg.angular_velocity.y  # 绕 Y 轴的角速度 → pitch 变化率
        gz = msg.angular_velocity.z  # 绕 Z 轴的角速度 → yaw 变化率

        # 首次调用: 用 orientation 初始化
        if not self.filter_initialized:
            if self.last_time is not None:
                self.roll_f = self.roll
                self.pitch_f = self.pitch
                self.yaw_f = self.yaw
                self.filter_initialized = True
                rospy.loginfo("[%s] 互补滤波器已初始化", self.node_name)
            return

        # 计算时间步长 dt (秒)
        dt = (msg.header.stamp - self.last_time).to_sec()

        # dt 合理性检查: 正常 ~0.001s (1000 Hz), 如果 dt ≤ 0 或 > 0.1s 说明有异常
        if dt <= 0.0 or dt > 0.1:
            rospy.logwarn_throttle(5.0,
                                   "[%s] dt 异常: %.4f s, 跳过本次积分",
                                   self.node_name, dt)
            return

        # ---- 步骤1: 用角速度积分预测新姿态 ----
        # roll 变化率并不简单地等于 gx, 因为欧拉角的导数与角速度之间
        # 存在运动学关系。但对于小角度 (< 30° pitch) 和短时间步长,
        # 近似: d_roll ≈ gx, d_pitch ≈ gy, d_yaw ≈ gz / cos(pitch)
        #
        # 精确的欧拉角运动学微分方程:
        #   d_roll  = gx + gy * sin(roll)*tan(pitch) + gz * cos(roll)*tan(pitch)
        #   d_pitch = gy * cos(roll) - gz * sin(roll)
        #   d_yaw   = gy * sin(roll)/cos(pitch) + gz * cos(roll)/cos(pitch)
        #
        # 这里使用精确公式以避免 pitch 接近 ±90° 时的奇异性

        sr = math.sin(self.roll_f)
        cr = math.cos(self.roll_f)
        sp = math.sin(self.pitch_f)
        cp = math.cos(self.pitch_f)

        # 避免除以零
        if abs(cp) < 1e-6:
            cp = 1e-6

        # 欧拉角导数 (精确运动学)
        d_roll = gx + gy * sr * sp / cp + gz * cr * sp / cp
        d_pitch = gy * cr - gz * sr
        d_yaw = gy * sr / cp + gz * cr / cp

        # 积分得到预测姿态
        pred_roll = self.roll_f + d_roll * dt
        pred_pitch = self.pitch_f + d_pitch * dt
        pred_yaw = self.yaw_f + d_yaw * dt

        # ---- 步骤2: 互补融合 ----
        # orientation(直接测量, 长期准) + gyro(积分预测, 短期快)
        self.roll_f = self.alpha * self.roll + (1.0 - self.alpha) * pred_roll
        self.pitch_f = self.alpha * self.pitch + (1.0 - self.alpha) * pred_pitch
        self.yaw_f = self.alpha * self.yaw + (1.0 - self.alpha) * pred_yaw

        # yaw 角度归一化到 [-π, π]
        self.yaw_f = math.atan2(math.sin(self.yaw_f), math.cos(self.yaw_f))

    # ================================================================
    # 发布与输出
    # ================================================================

    def _publish_attitude(self, msg):
        """
        发布姿态话题
        Vector3Stamped.x = roll (rad)
        Vector3Stamped.y = pitch (rad)
        Vector3Stamped.z = yaw (rad)
        """
        # 发布时间戳与 IMU 消息一致, 便于后续同步
        stamp = msg.header.stamp

        # 发布原始 / 基础姿态
        attitude_msg = Vector3Stamped()
        attitude_msg.header.stamp = stamp
        attitude_msg.header.frame_id = 'imu_link'
        attitude_msg.vector.x = self.roll
        attitude_msg.vector.y = self.pitch
        attitude_msg.vector.z = self.yaw
        self.attitude_pub.publish(attitude_msg)

        # 互补滤波模式下发布滤波后姿态
        if self.mode == 'complementary':
            filtered_msg = Vector3Stamped()
            filtered_msg.header.stamp = stamp
            filtered_msg.header.frame_id = 'imu_link'
            filtered_msg.vector.x = self.roll_f
            filtered_msg.vector.y = self.pitch_f
            filtered_msg.vector.z = self.yaw_f
            self.attitude_filtered_pub.publish(filtered_msg)

    def _print_status(self):
        """
        定时打印当前姿态到终端

        格式:
          [原始]  Roll:  +1.5°  Pitch:  -2.3°  Yaw: +45.0°
          [滤波]  Roll:  +1.4°  Pitch:  -2.2°  Yaw: +45.0°
          [真值]  Roll:  +1.5°  Pitch:  -2.3°  Yaw: +45.1°
        """
        # 弧度 → 度
        roll_deg = math.degrees(self.roll)
        pitch_deg = math.degrees(self.pitch)
        yaw_deg = math.degrees(self.yaw)

        # 构造输出字符串
        status_line = (
            f"[原始] Roll: {roll_deg:+7.2f}°  "
            f"Pitch: {pitch_deg:+7.2f}°  "
            f"Yaw: {yaw_deg:+7.2f}°"
        )

        if self.mode == 'complementary' and self.filter_initialized:
            roll_f_deg = math.degrees(self.roll_f)
            pitch_f_deg = math.degrees(self.pitch_f)
            yaw_f_deg = math.degrees(self.yaw_f)
            status_line += (
                f"\n[滤波] Roll: {roll_f_deg:+7.2f}°  "
                f"Pitch: {pitch_f_deg:+7.2f}°  "
                f"Yaw: {yaw_f_deg:+7.2f}°"
            )

        if self.use_ground_truth:
            gt_roll_deg = math.degrees(self.gt_roll)
            gt_pitch_deg = math.degrees(self.gt_pitch)
            gt_yaw_deg = math.degrees(self.gt_yaw)
            status_line += (
                f"\n[真值] Roll: {gt_roll_deg:+7.2f}°  "
                f"Pitch: {gt_pitch_deg:+7.2f}°  "
                f"Yaw: {gt_yaw_deg:+7.2f}°"
            )

        rospy.loginfo("\n%s", status_line)

    # ================================================================
    # 工具方法
    # ================================================================

    def get_attitude(self):
        """
        获取当前姿态 (外部调用接口)

        Returns:
            tuple (roll, pitch, yaw) 单位: 弧度
            互补滤波模式下返回滤波值, basic 模式返回原始值
        """
        if self.mode == 'complementary' and self.filter_initialized:
            return (self.roll_f, self.pitch_f, self.yaw_f)
        return (self.roll, self.pitch, self.yaw)


# ====================================================================
# 主函数
# ====================================================================

def main():
    """节点入口"""
    try:
        node = IMUAttitudeNode()
        # rospy.spin() 会阻塞在这里, 持续处理回调
        # 如果不需要在主线程做其他事, 这样最简洁
        rospy.spin()
    except rospy.ROSInterruptException:
        # 用户按 Ctrl+C 或节点被 kill 时触发
        rospy.loginfo("[imu_attitude_node] 节点正常退出")
    except KeyboardInterrupt:
        rospy.loginfo("[imu_attitude_node] 收到 Ctrl+C, 退出")


if __name__ == '__main__':
    main()
