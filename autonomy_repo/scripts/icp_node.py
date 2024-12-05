#!/usr/bin/env python3

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.time import Time

from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from sensor_msgs_py import point_cloud2
from tf2_ros import Buffer, TransformListener

from icp_utils import icp, open3d_icp


def get_pcd_array_from_point_cloud(pcd_msg: PointCloud2):
    pcd = point_cloud2.read_points_list(pcd_msg, field_names=["x", "y", "z"], skip_nans=True)
    return np.array(pcd)


def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    rotation_matrix = np.array([
        [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)]
    ])
    return rotation_matrix


class ICPNode(Node):
    def __init__(self):
        super().__init__('icp_node')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(0.2, self.get_tfs)

        self.prev_pcd = None
        self.pose = None
        self.lidar_pose = None
        self.transformation = np.eye(4)
        self.use_open3d = True
        
        # Task 2.3 — Subscribe to the point cloud
        self.pcd_sub = self.create_subscription(
            PointCloud2,
            '/velodyne_points',
            self.pcd_callback,
            10
        )
        
        # Task 2.4 — Subscribe to /odom
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
        
        self.icp_poses = []
        self.odom_poses = []

    def get_tfs(self):
        try:
            init_transform = self.tf_buffer.lookup_transform(target_frame='odom', source_frame='base_footprint', time=Time())
            
            translation = init_transform.transform.translation
            tx, ty, tz = translation.x, translation.y, translation.z
            
            rotation = init_transform.transform.rotation
            qx, qy, qz, qw = rotation.x, rotation.y, rotation.z, rotation.w
            rotation_matrix = quaternion_to_rotation_matrix(qx, qy, qz, qw)

            self.pose = np.eye(4)
            self.pose[:3, :3] = rotation_matrix
            self.pose[:3, 3] = [tx, ty, tz]
            self.get_logger().info('Set initial pose.')
            
            lidar_transform = self.tf_buffer.lookup_transform(target_frame='base_footprint', source_frame='velodyne', time=Time())
            
            lidar_translation = lidar_transform.transform.translation
            lx, ly, lz = lidar_translation.x, lidar_translation.y, lidar_translation.z

            lidar_rotation = lidar_transform.transform.rotation
            l_qx, l_qy, l_qz, l_qw = lidar_rotation.x, lidar_rotation.y, lidar_rotation.z, lidar_rotation.w
            lidar_rotation_matrix = quaternion_to_rotation_matrix(l_qx, l_qy, l_qz, l_qw)

            self.lidar_pose = np.eye(4)
            self.lidar_pose[:3, :3] = lidar_rotation_matrix
            self.lidar_pose[:3, 3] = [lx, ly, lz]
            self.get_logger().info('Set lidar pose.')
            
            self.timer.cancel()
            
        except Exception as e:
            self.pose = None
            self.lidar_pose = None

    def pcd_callback(self, msg: PointCloud2):
        if self.pose is None or self.lidar_pose is None:
            return
        
        if self.prev_pcd is None:
            pts = get_pcd_array_from_point_cloud(msg)
            self.prev_pcd = o3d.geometry.PointCloud()
            self.prev_pcd.points = o3d.utility.Vector3dVector(pts)
            self.prev_pcd = self.prev_pcd.uniform_down_sample(10)
            self.get_logger().info('Initial point cloud received.')
            return

        ### TODO: Task 2.5 ###
        pts = get_pcd_array_from_point_cloud(msg)
        current_pcd = o3d.geometry.PointCloud()
        current_pcd.points = o3d.utility.Vector3dVector(pts)
        current_pcd = current_pcd.uniform_down_sample(10)
        ### Task 2.5 ###

        ### TODO: Task 2.6 ###
        if self.use_open3d:
            # Use Open3D ICP method with PointCloud objects
            result = open3d_icp(
                source=self.prev_pcd,  # Previous point cloud
                target=current_pcd,   # Current point cloud
                T_init=self.transformation  # Initial guess
            )
            self.transformation = result  # Extract the transformation matrix
        else:
            # Use homework-defined ICP method with numpy arrays
            source_points = np.asarray(self.prev_pcd.points)  # Convert Open3D PCD to numpy
            target_points = np.asarray(current_pcd.points)    # Convert Open3D PCD to numpy
            
            # Call the homework-defined ICP function
            transformation, _ = icp(
                source_points=source_points, 
                target_points=target_points,
                init_transformation=self.transformation  # Initial guess
            )
            self.transformation = transformation
            
        ### Task 2.6 ###
        
        ### TODO: Task 2.7 ###
        self.pose = self.pose @ self.transformation
        # print(self.icp_poses)
        # print(type(self.icp_poses))
        self.icp_poses.append(self.pose[:2, 3])  # Store x, y for plotting
        ### Task 2.7 ###
        
        ### TODO: Task 2.8 ###
        self.prev_pcd = current_pcd
        ### Task 2.8 ###
        
    def odom_callback(self, msg: Odometry):
		### TODO: Task 2.9 ###
        pose = msg.pose.pose
        tx, ty = pose.position.x, pose.position.y
        self.odom_poses.append([tx, ty])

        ### Task 2.9 ###
        
    def plot_poses(self):
        icp_poses = np.array(self.icp_poses).reshape(-1, 2)
        print(icp_poses.shape)
        odom_poses = np.array(self.odom_poses)
        icp_info = 'Open3D' if self.use_open3d else 'HW3'

        # Plot x positions
        plt.figure(figsize=(8, 6))
        plt.plot(np.arange(len(icp_poses)), icp_poses[:, 0], 'b-', label='ICP x')
        plt.plot(np.arange(len(odom_poses)), odom_poses[:, 0], 'r-', label='ODOM x')
        plt.xlabel("N")
        plt.ylabel("X position")
        plt.legend()
        plt.title(f"X positions from {icp_info}")
        plt.grid()
        plt.savefig(Path(f"src/section/autonomy_repo/plots/{icp_info}_x.png"))

        # Plot y positions
        plt.figure(figsize=(8, 6))
        plt.plot(np.arange(len(icp_poses)), icp_poses[:, 1], 'b-', label='ICP y')
        plt.plot(np.arange(len(odom_poses)), odom_poses[:, 1], 'r-', label='ODOM y')
        plt.xlabel("N")
        plt.ylabel("Y position")
        plt.legend()
        plt.title(f"Y positions from {icp_info}")
        plt.grid()
        plt.savefig(Path(f"src/section/autonomy_repo/plots/{icp_info}_y.png"))


def main(args=None):
    rclpy.init(args=args)
    node = ICPNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.plot_poses()
        rclpy.shutdown()


if __name__ == '__main__':
    main()