#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
 FAST_LIO → 标准坐标系桥接节点 (Step 5 补充)
==============================================================================
 功能: 将 FAST_LIO 输出的里程计从私有坐标系桥接进项目的标准坐标系。

 背景:
   FAST_LIO 输出 nav_msgs/Odometry, frame_id = "camera_init",
   child_frame_id = "body"。而项目标准化 API 要求:
     odom_frame   = "odom"   (EKF 的 world_frame)
     base_frame   = "base"   (机器人本体)

   由于:
     - /cloud_base 已在 base 系 (cloud_preprocess 完成 laser_livox→base),
       FAST_LIO 外参为 identity, 故其 "body" == "base"
     - FAST_LIO 的 "camera_init" 初始化为起点, 与 "world"/"odom" 原点重合
   因此位置只需 frame 重命名 (identity)。

  ★ 航向对齐 (关键, 修复 EKF 发散):
     FAST_LIO 把初始化时的航向当作 0 (相对里程计, camera_init 系 yaw≈0),
     而 IMU /trunk_imu 给出 Gazebo 绝对航向 (spawn yaw=π/2, yaw≈1.566)。
     两者差约 90°。若直接把 camera_init 当 world 用, EKF 会同时融合两个
     矛盾的 yaw 观测 (0 vs 1.566) 导致状态发散。

     因此本节点把 FAST_LIO 的输出整体旋转到 world 系:
        - 旋转角 = 机器人初始航向 (默认自动取 /trunk_imu 首帧 yaw)
        - 位置 (x, y) 做 2D 旋转
        - 姿态四元数 q' = q_offset ⊗ q
        - twist 不处理: FAST_LIO 的 Odometry 未填充 twist (publish_odometry
          只 set pose), 且 EKF 的 odom0_config 已把 twist 全关

 订阅:
   /fast_lio_odom        nav_msgs/Odometry    FAST_LIO 原始里程计 (camera_init→body)
   /trunk_imu (可选)     sensor_msgs/Imu      仅用于自动对齐初始航向

 发布:
   /fast_lio_odom_base   nav_msgs/Odometry    标准坐标系里程计 (odom→base)
   odom → base TF        (可选, 当 EKF 不发布 TF 时开启)

 参数:
   ~odom_frame           str    odom 坐标系名称, 默认 "odom"
   ~base_frame           str    base 坐标系名称, 默认 "base"
   ~publish_tf           bool   是否广播 odom→base TF, 默认 false (由 EKF 负责)
   ~yaw_offset           float  显式指定航向偏移 (rad); 默认 None = 自动对齐 IMU
   ~imu_topic            str    自动对齐用的 IMU 话题, 默认 "/trunk_imu"

==============================================================================
"""

import math

import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import TransformStamped
import tf2_ros
from tf.transformations import (euler_from_quaternion,
                                quaternion_about_axis,
                                quaternion_multiply)


class FastLioBridge:
    """FAST_LIO 坐标系桥接节点"""

    def __init__(self):
        rospy.init_node('fastlio_bridge', anonymous=False)
        self.node_name = rospy.get_name()

        # ============================================================
        # 参数
        # ============================================================
        self.odom_frame = rospy.get_param('~odom_frame', 'odom')
        self.base_frame = rospy.get_param('~base_frame', 'base')
        self.publish_tf = str(rospy.get_param('~publish_tf', False)).lower() \
            in ('true', '1')

        # 航向对齐参数
        self.imu_topic = rospy.get_param('~imu_topic', '/trunk_imu')
        self.yaw_offset = rospy.get_param('~yaw_offset', None)

        # ============================================================
        # 发布
        # ============================================================
        self.odom_pub = rospy.Publisher('/fast_lio_odom_base', Odometry,
                                        queue_size=10)

        if self.publish_tf:
            self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        # ============================================================
        # 航向对齐初始化
        # ============================================================
        self._cos = 0.0
        self._sin = 0.0
        self._q_offset = None
        self._offset_ready = False

        if self.yaw_offset is not None:
            # 显式指定偏移, 无需订阅 IMU
            self._set_offset(self.yaw_offset)
        else:
            # 自动对齐: 订阅 IMU, 取首帧 yaw 作为偏移
            self._imu_sub = rospy.Subscriber(self.imu_topic, Imu,
                                             self._imu_callback, queue_size=1)
            rospy.loginfo("[%s] 等待首帧 IMU (%s) 以自动对齐航向...",
                          self.node_name, self.imu_topic)

        # ============================================================
        # 订阅
        # ============================================================
        rospy.Subscriber('/fast_lio_odom', Odometry,
                         self.odom_callback, queue_size=10)

        rospy.loginfo("[%s] FAST_LIO 桥接节点启动完成", self.node_name)
        rospy.loginfo("[%s] 订阅: /fast_lio_odom (camera_init→body)", self.node_name)
        rospy.loginfo("[%s] 发布: /fast_lio_odom_base (%s→%s), TF=%s",
                      self.node_name, self.odom_frame, self.base_frame,
                      self.publish_tf)

    # ================================================================
    # 航向对齐
    # ================================================================

    def _set_offset(self, offset):
        """设置航向偏移, 预计算旋转矩阵分量与偏移四元数"""
        self.yaw_offset = offset
        self._cos = math.cos(offset)
        self._sin = math.sin(offset)
        # 绕 Z 轴旋转 offset 的四元数 [x, y, z, w]
        self._q_offset = quaternion_about_axis(offset, (0.0, 0.0, 1.0))
        self._offset_ready = True
        rospy.loginfo("[%s] 航向对齐偏移 = %.4f rad (%.2f°)",
                      self.node_name, offset, math.degrees(offset))

    def _imu_callback(self, msg):
        """取首帧 IMU 航向作为对齐偏移, 然后锁定 (不再跟随机器人转动)"""
        if self._offset_ready:
            return
        (_, _, yaw) = euler_from_quaternion(
            [msg.orientation.x, msg.orientation.y,
             msg.orientation.z, msg.orientation.w])
        self._set_offset(yaw)
        self._imu_sub.unregister()

    # ================================================================
    # 里程计回调
    # ================================================================

    def odom_callback(self, msg):
        """接收 FAST_LIO 里程计, 旋转到 world 系并重命名 frame 后转发"""
        if not self._offset_ready:
            rospy.logwarn_throttle(
                5.0, "[%s] 航向偏移未就绪, 丢弃 /fast_lio_odom", self.node_name)
            return

        self._rotate_pose(msg.pose.pose)

        # 重命名: camera_init → odom, body → base
        msg.header.frame_id = self.odom_frame
        msg.child_frame_id = self.base_frame

        self.odom_pub.publish(msg)

        if self.publish_tf:
            self._broadcast_tf(msg)

    def _rotate_pose(self, pose):
        """把 pose (position + orientation) 旋转 yaw_offset 到 world 系

        位置:  2D 旋转
          x' = cos·x - sin·y
          y' = sin·x + cos·y
        姿态:  q' = q_offset ⊗ q

        twist 不旋转: FAST_LIO 未填充 twist (恒为零), 且 EKF 不融合 odom0 twist。
        """
        c, s = self._cos, self._sin

        # 位置 (2D 旋转, z 不变)
        x, y = pose.position.x, pose.position.y
        pose.position.x = c * x - s * y
        pose.position.y = s * x + c * y

        # 姿态 (四元数左乘偏移)
        q = pose.orientation
        q_rot = quaternion_multiply(self._q_offset, [q.x, q.y, q.z, q.w])
        pose.orientation.x = q_rot[0]
        pose.orientation.y = q_rot[1]
        pose.orientation.z = q_rot[2]
        pose.orientation.w = q_rot[3]

    # ================================================================
    # TF 广播
    # ================================================================

    def _broadcast_tf(self, odom):
        """广播 odom → base TF (仅当 EKF 不负责时使用)"""
        t = TransformStamped()
        t.header.stamp = odom.header.stamp
        t.header.frame_id = self.odom_frame
        t.child_frame_id = self.base_frame

        p = odom.pose.pose.position
        q = odom.pose.pose.orientation
        t.transform.translation.x = p.x
        t.transform.translation.y = p.y
        t.transform.translation.z = p.z
        t.transform.rotation.x = q.x
        t.transform.rotation.y = q.y
        t.transform.rotation.z = q.z
        t.transform.rotation.w = q.w

        self.tf_broadcaster.sendTransform(t)


def main():
    try:
        node = FastLioBridge()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("[fastlio_bridge] 节点正常退出")
    except KeyboardInterrupt:
        rospy.loginfo("[fastlio_bridge] 收到 Ctrl+C, 退出")


if __name__ == '__main__':
    main()
