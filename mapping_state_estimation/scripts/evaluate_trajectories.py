#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
 轨迹误差评估脚本
==============================================================================
 读取 TUM 格式轨迹文件，计算 ATE (绝对轨迹误差) 和 RPE (相对位姿误差)，
 生成误差图表和统计报告。

 使用方式:
   # 评估 EKF vs 真值
   python3 evaluate_trajectories.py ground_truth.txt ekf_filtered.txt

   # 同时评估 EKF 和 LiDAR里程计
   python3 evaluate_trajectories.py ground_truth.txt ekf_filtered.txt lidar_odom.txt

   # 指定输出目录
   python3 evaluate_trajectories.py -o /tmp/results ground_truth.txt ekf_filtered.txt

 依赖: numpy, matplotlib (pip3 install numpy matplotlib)
==============================================================================
"""

import argparse
import math
import os
import sys
import numpy as np

# --- 轨迹读取 -----------------------------------------------------------

def read_tum_trajectory(filepath):
    """
    读取 TUM 格式轨迹文件, 返回:
        stamps:  (N,)  float64 时间戳数组
        poses:   (N,7) float64 [tx, ty, tz, qx, qy, qz, qw]
    """
    stamps = []
    poses = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            stamps.append(float(parts[0]))
            poses.append([float(x) for x in parts[1:8]])

    if len(stamps) == 0:
        raise ValueError(f"文件 {filepath} 没有有效数据行")

    return np.array(stamps, dtype=np.float64), np.array(poses, dtype=np.float64)


# --- SE(3) 数学工具 -----------------------------------------------------

def quat_to_rotm(qx, qy, qz, qw):
    """四元数 → 3x3 旋转矩阵"""
    R = np.zeros((3, 3), dtype=np.float64)
    R[0, 0] = 1 - 2*qy*qy - 2*qz*qz
    R[0, 1] = 2*qx*qy - 2*qz*qw
    R[0, 2] = 2*qx*qz + 2*qy*qw
    R[1, 0] = 2*qx*qy + 2*qz*qw
    R[1, 1] = 1 - 2*qx*qx - 2*qz*qz
    R[1, 2] = 2*qy*qz - 2*qx*qw
    R[2, 0] = 2*qx*qz - 2*qy*qw
    R[2, 1] = 2*qy*qz + 2*qx*qw
    R[2, 2] = 1 - 2*qx*qx - 2*qy*qy
    return R

def pose_to_T(pose):
    """[tx,ty,tz,qx,qy,qz,qw] → 4x4 变换矩阵"""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_to_rotm(*pose[3:7])
    T[:3, 3] = pose[0:3]
    return T

def T_error(T_est, T_gt):
    """计算两个变换矩阵之间的误差 (平移m, 旋转rad)"""
    # 平移误差: ||t_est - t_gt||
    trans_err = np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3])

    # 旋转误差: 从相对旋转矩阵中提取旋转角
    R_rel = T_est[:3, :3].T @ T_gt[:3, :3]
    trace = np.trace(R_rel)
    cos_angle = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
    rot_err = math.acos(cos_angle)

    return trans_err, rot_err


# --- 时间对齐 -----------------------------------------------------------

def associate_trajectories(stamps_est, poses_est, stamps_gt, poses_gt, max_diff=0.1):
    """
    按时间戳关联两条轨迹, 返回对齐后的位姿列表。
    max_diff: 最大允许时间差 (秒), 超过此值不匹配
    """
    matches_est = []
    matches_gt = []

    j = 0  # ground truth 索引
    for i in range(len(stamps_est)):
        t_est = stamps_est[i]
        # 在 gt 中找最近的时间戳
        while j < len(stamps_gt) and stamps_gt[j] < t_est - max_diff:
            j += 1
        if j >= len(stamps_gt):
            break
        if abs(stamps_gt[j] - t_est) <= max_diff:
            matches_est.append(poses_est[i])
            matches_gt.append(poses_gt[j])

    if len(matches_est) == 0:
        raise RuntimeError(
            f"无法对齐轨迹! 最大时间差={max_diff}s, "
            f"est 时间范围=[{stamps_est[0]:.2f}, {stamps_est[-1]:.2f}], "
            f"gt 时间范围=[{stamps_gt[0]:.2f}, {stamps_gt[-1]:.2f}]")

    return np.array(matches_est), np.array(matches_gt)


# --- ATE: 绝对轨迹误差 --------------------------------------------------

def compute_ate(poses_est, poses_gt):
    """
    计算绝对轨迹误差 (Absolute Trajectory Error)

    先通过 Umeyama 对齐两条轨迹 (消除坐标系偏移),
    然后计算逐点欧氏距离作为 ATE。

    Returns:
        ate_trans: (N,) 逐点平移误差 (m)
        ate_rot:   (N,) 逐点旋转误差 (rad)
        stats:     dict 统计信息
    """
    n = len(poses_est)

    # 中心化
    t_est = poses_est[:, :3]
    t_gt = poses_gt[:, :3]

    mu_est = t_est.mean(axis=0)
    mu_gt = t_gt.mean(axis=0)

    t_est_c = t_est - mu_est
    t_gt_c = t_gt - mu_gt

    # SVD 求最优旋转
    H = t_est_c.T @ t_gt_c
    U, S, Vt = np.linalg.svd(H)
    R_align = Vt.T @ U.T

    if np.linalg.det(R_align) < 0:
        Vt[2, :] *= -1
        R_align = Vt.T @ U.T

    # 对齐后的估计轨迹
    t_est_aligned = (R_align @ t_est_c.T).T + mu_gt

    # 逐点误差
    ate_trans = np.linalg.norm(t_est_aligned - t_gt, axis=1)

    # 旋转误差: 先对齐旋转部分, 再逐帧计算
    ate_rot = np.zeros(n)
    for i in range(n):
        T_est = pose_to_T(poses_est[i])
        T_gt = pose_to_T(poses_gt[i])
        T_est[:3, :3] = R_align @ T_est[:3, :3]
        _, ate_rot[i] = T_error(T_est, T_gt)

    stats = {
        'rmse':       float(np.sqrt(np.mean(ate_trans ** 2))),
        'mean':       float(np.mean(ate_trans)),
        'median':     float(np.median(ate_trans)),
        'std':        float(np.std(ate_trans)),
        'min':        float(np.min(ate_trans)),
        'max':        float(np.max(ate_trans)),
        'rmse_deg':   float(np.sqrt(np.mean(ate_rot ** 2))) * 180.0 / math.pi,
        'mean_deg':   float(np.mean(ate_rot)) * 180.0 / math.pi,
        'n_points':   n,
    }

    return ate_trans, ate_rot, stats


# --- RPE: 相对位姿误差 --------------------------------------------------

def compute_rpe(poses_est, poses_gt, delta=1):
    """
    计算相对位姿误差 (Relative Pose Error)

    比较相邻 delta 帧之间的相对运动量。反映局部漂移。

    Returns:
        rpe_trans: (N-delta,) 逐段平移误差 (m)
        rpe_rot:   (N-delta,) 逐段旋转误差 (rad)
        stats:     dict
    """
    n = len(poses_est)
    errors_trans = []
    errors_rot = []

    for i in range(n - delta):
        j = i + delta

        # 真值增量
        T_gt_i = pose_to_T(poses_gt[i])
        T_gt_j = pose_to_T(poses_gt[j])
        T_gt_rel = np.linalg.inv(T_gt_i) @ T_gt_j

        # 估计增量
        T_est_i = pose_to_T(poses_est[i])
        T_est_j = pose_to_T(poses_est[j])
        T_est_rel = np.linalg.inv(T_est_i) @ T_est_j

        # 误差
        trans_err, rot_err = T_error(T_est_rel, T_gt_rel)
        errors_trans.append(trans_err)
        errors_rot.append(rot_err)

    errors_trans = np.array(errors_trans)
    errors_rot = np.array(errors_rot)

    stats = {
        'delta':      delta,
        'rmse':       float(np.sqrt(np.mean(errors_trans ** 2))),
        'mean':       float(np.mean(errors_trans)),
        'median':     float(np.median(errors_trans)),
        'std':        float(np.std(errors_trans)),
        'min':        float(np.min(errors_trans)),
        'max':        float(np.max(errors_trans)),
        'rmse_deg':   float(np.sqrt(np.mean(errors_rot ** 2))) * 180.0 / math.pi,
        'mean_deg':   float(np.mean(errors_rot)) * 180.0 / math.pi,
        'n_points':   len(errors_trans),
    }

    return errors_trans, errors_rot, stats


# --- 绘图 ---------------------------------------------------------------

def plot_results(ate_trans, ate_rot, rpe_trans, rpe_rot,
                 est_label, output_dir):
    """生成误差图并保存"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Trajectory Error: {est_label}', fontsize=14)

    # ATE 平移
    ax = axes[0, 0]
    ax.plot(ate_trans, alpha=0.7, linewidth=0.5)
    ax.axhline(y=np.mean(ate_trans), color='r', linestyle='--',
               label=f'mean={np.mean(ate_trans):.3f}m')
    ax.set_title('ATE Translation Error')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Error (m)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ATE 旋转
    ax = axes[0, 1]
    ax.plot(np.degrees(ate_rot), alpha=0.7, linewidth=0.5, color='orange')
    ax.axhline(y=np.degrees(np.mean(ate_rot)), color='r', linestyle='--',
               label=f'mean={np.degrees(np.mean(ate_rot)):.3f}°')
    ax.set_title('ATE Rotation Error')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Error (deg)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # RPE 平移
    ax = axes[1, 0]
    ax.plot(rpe_trans, alpha=0.7, linewidth=0.5, color='green')
    ax.axhline(y=np.mean(rpe_trans), color='r', linestyle='--',
               label=f'mean={np.mean(rpe_trans):.4f}m/frame')
    ax.set_title('RPE Translation Error (frame-to-frame)')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Error (m)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # RPE 旋转
    ax = axes[1, 1]
    ax.plot(np.degrees(rpe_rot), alpha=0.7, linewidth=0.5, color='brown')
    ax.axhline(y=np.degrees(np.mean(rpe_rot)), color='r', linestyle='--',
               label=f'mean={np.degrees(np.mean(rpe_rot)):.3f}°/frame')
    ax.set_title('RPE Rotation Error (frame-to-frame)')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Error (deg)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fpath = os.path.join(output_dir, f'{est_label}_error_plot.png')
    plt.savefig(fpath, dpi=150)
    plt.close()
    print(f"[evaluate] 已保存图表: {fpath}")


# --- 主函数 -------------------------------------------------------------

def evaluate_one(gt_file, est_file, label, output_dir, rpe_delta=1):
    """评估单条估计轨迹与真值的误差"""
    print(f"\n{'='*60}")
    print(f"[evaluate] 评估: {label}")
    print(f"[evaluate]   真值: {gt_file}")
    print(f"[evaluate]   估计: {est_file}")
    print(f"{'='*60}")

    stamps_gt, poses_gt = read_tum_trajectory(gt_file)
    stamps_est, poses_est = read_tum_trajectory(est_file)

    print(f"[evaluate] 真值轨迹: {len(stamps_gt)} 帧, "
          f"时长 {stamps_gt[-1]-stamps_gt[0]:.1f}s")
    print(f"[evaluate] 估计轨迹: {len(stamps_est)} 帧, "
          f"时长 {stamps_est[-1]-stamps_est[0]:.1f}s")

    # 时间对齐
    poses_est_a, poses_gt_a = associate_trajectories(
        stamps_est, poses_est, stamps_gt, poses_gt)
    print(f"[evaluate] 对齐后: {len(poses_est_a)} 对匹配点")

    # ATE
    ate_trans, ate_rot, ate_stats = compute_ate(poses_est_a, poses_gt_a)

    # RPE
    rpe_trans, rpe_rot, rpe_stats = compute_rpe(
        poses_est_a, poses_gt_a, delta=rpe_delta)

    # 打印报告
    print(f"\n{'─'*50}")
    print(f"  ATE (绝对轨迹误差)")
    print(f"{'─'*50}")
    print(f"  平移 RMSE:  {ate_stats['rmse']:.4f} m")
    print(f"  平移 Mean:  {ate_stats['mean']:.4f} m")
    print(f"  平移 Std:   {ate_stats['std']:.4f} m")
    print(f"  平移 Max:   {ate_stats['max']:.4f} m")
    print(f"  旋转 RMSE:  {ate_stats['rmse_deg']:.4f} °")
    print(f"  旋转 Mean:  {ate_stats['mean_deg']:.4f} °")

    print(f"\n{'─'*50}")
    print(f"  RPE (相对位姿误差, delta={rpe_delta})")
    print(f"{'─'*50}")
    print(f"  平移 RMSE:  {rpe_stats['rmse']:.4f} m/frame")
    print(f"  平移 Mean:  {rpe_stats['mean']:.4f} m/frame")
    print(f"  旋转 RMSE:  {rpe_stats['rmse_deg']:.4f} °/frame")
    print(f"  旋转 Mean:  {rpe_stats['mean_deg']:.4f} °/frame")

    # 保存统计到文件
    stats_file = os.path.join(output_dir, f'{label}_stats.txt')
    with open(stats_file, 'w') as f:
        f.write(f"=== {label} vs Ground Truth ===\n\n")
        f.write("ATE (Absolute Trajectory Error):\n")
        for k, v in ate_stats.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nRPE (Relative Pose Error, delta={rpe_delta}):\n")
        for k, v in rpe_stats.items():
            f.write(f"  {k}: {v}\n")
    print(f"\n[evaluate] 统计已保存: {stats_file}")

    # 保存对齐后的轨迹 (用于外部工具绘图)
    traj_file = os.path.join(output_dir, f'{label}_aligned.txt')
    with open(traj_file, 'w') as f:
        f.write('# timestamp_gt tx_gt ty_gt tz_gt qx_gt qy_gt qz_gt qw_gt '
                'tx_est ty_est tz_est qx_est qy_est qz_est qw_est '
                'ate_trans ate_rot_deg\n')
        for i, gt_pose in enumerate(poses_gt_a):
            est_pose = poses_est_a[i]
            f.write(f"{stamps_gt[i]:.6f} "
                    f"{' '.join(f'{x:.6f}' for x in gt_pose)} "
                    f"{' '.join(f'{x:.6f}' for x in est_pose)} "
                    f"{ate_trans[i]:.6f} {ate_rot[i]*180/math.pi:.6f}\n")
    print(f"[evaluate] 对齐轨迹已保存: {traj_file}")

    # 绘图
    try:
        plot_results(ate_trans, ate_rot, rpe_trans, rpe_rot,
                     label, output_dir)
    except Exception as e:
        print(f"[evaluate] 绘图失败 (可忽略): {e}")

    # 返回简要结果
    return {
        'label': label,
        'ate_rmse': ate_stats['rmse'],
        'ate_rmse_deg': ate_stats['rmse_deg'],
        'rpe_rmse': rpe_stats['rmse'],
        'rpe_rmse_deg': rpe_stats['rmse_deg'],
    }


def main():
    parser = argparse.ArgumentParser(
        description='评估 SLAM 轨迹误差 (ATE + RPE)')
    parser.add_argument('gt_file', help='地面真值轨迹 (TUM格式)')
    parser.add_argument('est_files', nargs='+',
                        help='估计轨迹文件, 可多个')
    parser.add_argument('-o', '--output_dir', default=None,
                        help='输出目录 (默认在第一个估计文件旁边建 results/)')
    parser.add_argument('-d', '--rpe_delta', type=int, default=1,
                        help='RPE 帧间隔 (默认1 = 逐帧)')
    parser.add_argument('--labels', nargs='*', default=None,
                        help='每个估计轨迹的标签 (与文件一一对应)')
    args = parser.parse_args()

    # 输出目录
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(os.path.dirname(args.est_files[0]), 'results')
    os.makedirs(output_dir, exist_ok=True)

    # 标签
    labels = args.labels if args.labels else [
        os.path.splitext(os.path.basename(f))[0] for f in args.est_files
    ]

    # 逐个评估
    summary = []
    for est_file, label in zip(args.est_files, labels):
        result = evaluate_one(args.gt_file, est_file, label, output_dir,
                              rpe_delta=args.rpe_delta)
        summary.append(result)

    # 最终汇总
    print(f"\n{'='*60}")
    print(f"  汇总对比")
    print(f"{'='*60}")
    print(f"  {'方法':<25s} {'ATE平移':>10s} {'ATE旋转':>10s} {'RPE平移':>12s} {'RPE旋转':>12s}")
    print(f"  {'─'*69}")
    for r in summary:
        print(f"  {r['label']:<25s} {r['ate_rmse']:8.4f}m  "
              f"{r['ate_rmse_deg']:8.4f}°  "
              f"{r['rpe_rmse']:10.4f}m  {r['rpe_rmse_deg']:10.4f}°")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
