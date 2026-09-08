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
   因此本节点只需做 frame 重命名 (identity), 不做任何位姿变换。

 订阅:
   /fast_lio_odom        nav_msgs/Odometry    FAST_LIO 原始里程计 (camera_init→body)

 发布:
   /fast_lio_odom_base   nav_msgs/Odometry    标准坐标系里程计 (odom→base)
   odom → base TF        (可选, 当 EKF 不发布 TF 时开启)

 参数:
   ~odom_frame           str    odom 坐标系名称, 默认 "odom"
   ~base_frame           str    base 坐标系名称, 默认 "base"
   ~publish_tf           bool   是否广播 odom→base TF, 默认 false (由 EKF 负责)

==============================================================================
"""

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
import tf2_ros


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

        # ============================================================
        # 发布
        # ============================================================
        self.odom_pub = rospy.Publisher('/fast_lio_odom_base', Odometry,
                                        queue_size=10)

        if self.publish_tf:
            self.tf_broadcaster = tf2_ros.TransformBroadcaster()

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

    def odom_callback(self, msg):
        """接收 FAST_LIO 里程计, 重命名 frame 后转发"""
        # identity 重命名: camera_init → odom, body → base
        msg.header.frame_id = self.odom_frame
        msg.child_frame_id = self.base_frame

        self.odom_pub.publish(msg)

        if self.publish_tf:
            self._broadcast_tf(msg)

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
