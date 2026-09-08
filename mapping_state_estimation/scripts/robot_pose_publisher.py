#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
 机器人位姿发布节点 (Step 10)
==============================================================================
 功能: 从里程计获取当前位姿，发布 /robot_pose 供目标识别模块使用。

 订阅:
   /odometry/filtered  nav_msgs/Odometry    融合里程计 (优先)
   /lidar_odom         nav_msgs/Odometry    LiDAR 里程计 (回退)

 发布:
   /robot_pose         geometry_msgs/PoseStamped  当前机器人位姿 (world 系)

==============================================================================
"""

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped


class RobotPosePublisher:
    """机器人位姿发布器"""

    def __init__(self):
        rospy.init_node('robot_pose_publisher', anonymous=False)
        self.node_name = rospy.get_name()

        self.pose_pub = rospy.Publisher('/robot_pose', PoseStamped,
                                         queue_size=10)

        # 订阅里程计
        rospy.Subscriber('/odometry/filtered', Odometry,
                         self.odom_callback, queue_size=10)
        rospy.Subscriber('/lidar_odom', Odometry,
                         self.odom_callback, queue_size=10)

        self.last_filtered_time = rospy.Time(0)

        rospy.loginfo("[%s] 机器人位姿发布节点启动完成", self.node_name)

    def odom_callback(self, msg):
        """里程计回调: 转发为 PoseStamped"""
        # 优先使用 /odometry/filtered
        topic = msg._connection_header['topic']
        if topic == '/lidar_odom':
            # 只在没有 EKF 时使用 lidar_odom
            if (rospy.Time.now() - self.last_filtered_time).to_sec() < 1.0:
                return

        if topic == '/odometry/filtered':
            self.last_filtered_time = msg.header.stamp

        pose_msg = PoseStamped()
        pose_msg.header.stamp = msg.header.stamp
        pose_msg.header.frame_id = 'world'
        pose_msg.pose = msg.pose.pose

        self.pose_pub.publish(pose_msg)


def main():
    try:
        node = RobotPosePublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
