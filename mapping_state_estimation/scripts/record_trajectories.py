#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
 轨迹录制节点
==============================================================================
 同时录制三条轨迹到文本文件 (TUM 格式, 兼容 evo 评估工具):
   1. 地面真值: /Odometry_gazebo → ground_truth.txt
   2. EKF 融合:  /odometry/filtered → ekf_filtered.txt
   3. LiDAR里程计: /lidar_odom      → lidar_odom.txt

 TUM 格式每行: timestamp tx ty tz qx qy qz qw

 使用方式:
   rosrun mapping_state_estimation record_trajectories.py
   rosrun mapping_state_estimation record_trajectories.py _output_dir:=/tmp/eval1
==============================================================================
"""

import os
import sys
import rospy
from nav_msgs.msg import Odometry


class TrajectoryRecorder:
    """录制多条里程计轨迹"""

    def __init__(self):
        rospy.init_node('trajectory_recorder', anonymous=False)

        # 输出目录
        self.output_dir = rospy.get_param('~output_dir',
                                          os.path.join(os.path.expanduser('~'),
                                                       'trajectory_eval'))
        os.makedirs(self.output_dir, exist_ok=True)

        rospy.loginfo("[recorder] 输出目录: %s", self.output_dir)

        # 打开三个文件
        self.files = {}
        topics = {
            '/Odometry_gazebo':    'ground_truth.txt',
            '/odometry/filtered':  'ekf_filtered.txt',
            '/lidar_odom':         'lidar_odom.txt',
        }

        self.counts = {}

        for topic, fname in topics.items():
            fpath = os.path.join(self.output_dir, fname)
            self.files[topic] = open(fpath, 'w')
            # TUM 格式表头注释
            self.files[topic].write(
                '# timestamp tx ty tz qx qy qz qw\n')
            self.counts[topic] = 0
            # 订阅
            rospy.Subscriber(topic, Odometry,
                             self._make_callback(topic), queue_size=100)
            rospy.loginfo("[recorder] 订阅: %s → %s", topic, fpath)

        self.print_interval = rospy.get_param('~print_interval', 5.0)
        self.last_print = rospy.Time.now()

        rospy.loginfo("[recorder] 轨迹录制中... 按 Ctrl+C 停止并保存")

    def _make_callback(self, topic):
        """生成回调闭包"""
        def callback(msg):
            t = msg.header.stamp.to_sec()
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            line = f"{t:.6f} {p.x:.6f} {p.y:.6f} {p.z:.6f} {q.x:.6f} {q.y:.6f} {q.z:.6f} {q.w:.6f}\n"
            self.files[topic].write(line)
            self.counts[topic] += 1

            now = rospy.Time.now()
            if (now - self.last_print).to_sec() > self.print_interval:
                rospy.loginfo("[recorder] 已录制: GT=%d  EKF=%d  LidarOdom=%d",
                              self.counts['/Odometry_gazebo'],
                              self.counts['/odometry/filtered'],
                              self.counts['/lidar_odom'])
                self.last_print = now

        return callback

    def close(self):
        for topic, f in self.files.items():
            f.close()
            rospy.loginfo("[recorder] 已保存: %s (%d 条)",
                          os.path.join(self.output_dir,
                                       os.path.basename(f.name)),
                          self.counts[topic])


def main():
    recorder = TrajectoryRecorder()
    try:
        rospy.spin()
    except (rospy.ROSInterruptException, KeyboardInterrupt):
        rospy.loginfo("[recorder] 正在保存文件...")
    finally:
        recorder.close()
        rospy.loginfo("[recorder] 完成! 文件保存在 %s", recorder.output_dir)


if __name__ == '__main__':
    main()
