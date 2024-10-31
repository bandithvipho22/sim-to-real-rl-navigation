#!/usr/bin/env python3

from math import atan2, hypot, sqrt, pi
import time
import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import Quaternion, TransformStamped, Vector3Stamped, Pose2D
from std_msgs.msg import Float64
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseWithCovarianceStamped
from turtlesim.msg import Pose



def quaternion_from_euler(roll, pitch, yaw) -> Quaternion:
    cy = math.cos(yaw*0.5)
    sy = math.sin(yaw*0.5)
    cp = math.cos(pitch*0.5)
    sp = math.sin(pitch*0.5)
    cr = math.cos(roll*0.5)
    sr = math.sin(roll*0.5)

    q = Quaternion()
    q.w = cy * cp * cr + sy * sp * sr
    q.x = cy * cp * sr - sy * sp * cr
    q.y = sy * cp * sr + cy * sp * cr
    q.z = sy * cp * cr - cy * sp * sr

    return q

class OdomPublisher(Node):
    def __init__(self):
        super().__init__('odometry_publisher')
        self.position_subscription = self.create_subscription(
            Vector3Stamped,
            'position',
            self.position_callback,
            10
        )
        
        self.imu_subscription = self.create_subscription(
            Imu,
            'imu/data_raw',
            self.imu_callback,
            10
        )
        
        self.yaw_publisher = self.create_publisher(
            Float64,
            'yaw',
            10
        )
        
        self.odom_publisher = self.create_publisher(
            Odometry,
            '/demo/odom',#'odom',
            10
        )

        self.Pose_Estimate_subscription = self.create_subscription(PoseWithCovarianceStamped, '/Pose_Estimate', self.pose_estimate_callback,1)
        
        self.tf_broadcaster = TransformBroadcaster(self)
        
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.Ts = 0.01
        self.currentTime = time.perf_counter()
        self.prevTime = time.perf_counter() - 0.01
        
        self.old_x = 0.0
        self.old_y = 0.0
        self.old_theta = 0.0
        
        self.home_x = 0.0
        self.home_y = 0.0
        self.home_position_set = False
        
        self.home_theta = 0.0
        self.home_orientation_set = False

        self.curr_pose = Pose()

    
    def imu_callback(self, msg):
        
        qw = msg.orientation.w
        qx = msg.orientation.x
        qy = msg.orientation.y
        qz = msg.orientation.z
        
        yaw_read = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)) 
        print(yaw_read)
        if self.home_orientation_set == False:
            self.home_theta = yaw_read
            self.home_orientation_set = True

        theta = yaw_read - self.home_theta
        
        if theta > math.pi:
            theta -= 2 * math.pi
        elif theta < -math.pi:
            theta += 2 * math.pi
            
        yaw_msg = Float64()
        yaw_msg.data = theta
        
        self.yaw_publisher.publish(yaw_msg)
        
    def position_callback(self, msg):
        if self.home_position_set == False:
            self.home_x = msg.vector.x
            self.home_y = msg.vector.y
            self.home_position_set = True
            
        self.currentTime = time.perf_counter()
        Ts = self.currentTime - self.prevTime
        self.prevTime = self.currentTime

        self.x = msg.vector.x - self.home_x 
        self.y = msg.vector.y - self.home_y 
        self.theta = msg.vector.z
        
        self.vx = (self.x - self.old_x)/Ts
        self.vy = (self.y - self.old_y) /Ts
        self.old_x = self.x
        self.old_y = self.y
        
        self.wz = (self.theta - self.old_theta) / Ts
        self.old_theta = self.theta
        
        robot_orientation = quaternion_from_euler(0, 0, self.theta)
        timestamp = self.get_clock().now().to_msg() 
        t = TransformStamped()

        t.header.stamp = timestamp 
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        # t.transform.translation.z = 0.0
        t.transform.rotation = robot_orientation

        odom_msg = Odometry()
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'base_link'
        odom_msg.header.stamp = self.get_clock().now().to_msg() 
        odom_msg.pose.pose.position.x = self.x
        odom_msg.pose.pose.position.y =  self.y
        odom_msg.pose.pose.position.z = 0.0
        odom_msg.pose.pose.orientation = robot_orientation
        odom_msg.twist.twist.linear.x = self.vx
        odom_msg.twist.twist.linear.y = self.vy
        odom_msg.twist.twist.angular.z = self.wz
        self.tf_broadcaster.sendTransform(t)
        self.odom_publisher.publish(odom_msg)
        self.get_logger().info('x = %f, y = : %f, theta = %f' % (self.x, self.y, self.theta))
        
    def pose_estimate_callback(self, initial_pose):
        """Determine whether to accept or reject the goal"""
        
        
        self.curr_pose.x = initial_pose.pose.pose.position.x
        self.curr_pose.y = initial_pose.pose.pose.position.y
        
        qw = initial_pose.pose.pose.orientation.w
        qx = initial_pose.pose.pose.orientation.x
        qy = initial_pose.pose.pose.orientation.y
        qz = initial_pose.pose.pose.orientation.z

        self.curr_pose.theta = atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        #print(self.curr_pose.x)

def main(args=None):
    rclpy.init(args=args)
    robot_control_node = OdomPublisher()
    rclpy.spin(robot_control_node)

    robot_control_node.destroy_node()
    rclpy.shutdown()
if __name__ == '__main__':
    main()