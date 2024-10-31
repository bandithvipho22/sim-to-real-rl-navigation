import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PoseStamped 
from nav_msgs.msg import Path 
from std_srvs.srv import Empty
from functools import partial
import numpy as np
import math
from gazebo_msgs.srv import DeleteEntity, SpawnEntity, SetModelState, SetEntityState
from std_msgs.msg import String
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point 

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')
        self.get_logger().info("The robot controller node has been created")

        # Existing publishers and subscribers
        self.action_pub = self.create_publisher(Twist, '/demo/cmd_vel', 10)
        self.pose_sub = self.create_subscription(Odometry, '/demo/odom', self.odom_callback, 1)
        self.laser_sub = self.create_subscription(LaserScan, '/demo/laser/out', self.laser_callback, 1)
        self.client_state = self.create_client(SetEntityState, "/demo/set_entity_state")
        self.path_pub = self.create_publisher(Path, '/robot_path', 10)
        self.action_type_pub = self.create_publisher(String, '/agent_action_type', 10)
        self.marker_publisher = self.create_publisher(Marker, '/marker', 10)

        # Initialize attributes
        self._agent_location = np.array([np.float32(1), np.float32(16)])
        self._agent_orientation = 0.0
        self._laser_reads = np.array([np.float32(10)] * 61)
        self._target_location = np.array([np.float32(1), np.float32(10)])
        self.path = Path()
        self.path.header.frame_id = "odom"
        self.marker_id = 0

        # Pure Pursuit parameters
        self.lookahead_distance = 1.0 #0.5
        self.kp_v = 2.0  # Gain for linear velocity
        self.max_linear_velocity = 0.40
        self.max_angular_velocity = 0.35 #0.5

        # Blend factor for RL and Pure Pursuit (0: full RL, 1: full Pure Pursuit)
        self.pp_balance_factor = 0.6

       
    def send_velocity_command(self, velocity):
        msg = Twist()
        msg.linear.x = float(velocity[0])
        msg.angular.z = float(velocity[1])
        self.action_pub.publish(msg)

    def odom_callback(self, msg: Odometry):
        self._agent_location = np.array([np.float32(np.clip(msg.pose.pose.position.x, -12, 12)),
                                         np.float32(np.clip(msg.pose.pose.position.y, -35, 21))])
        self._agent_orientation = 2 * math.atan2(msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)
        self._done_pose = True
        self.update_and_publish_path(msg)

    def laser_callback(self, msg: LaserScan):
        self._laser_reads = np.array(msg.ranges)
        self._laser_reads[self._laser_reads == np.inf] = np.float32(10)
        self._done_laser = True

    def update_and_publish_path(self, msg: Odometry):
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = "odom"
        pose.pose = msg.pose.pose
        self.path.poses.append(pose)
        self.path_pub.publish(self.path)

    def reset_path(self):
        self.path.poses = []

    def pure_pursuit_control(self):
        dx = self._target_location[0] - self._agent_location[0]
        dy = self._target_location[1] - self._agent_location[1]
        
        target_angle = math.atan2(dy, dx)
        alpha = target_angle - self._agent_orientation
        
        # Normalize alpha to [-pi, pi]
        alpha = (alpha + math.pi) % (2 * math.pi) - math.pi

        # Calculate curvature
        ld = self.lookahead_distance
        curvature = 2 * math.sin(alpha) / ld

        # Calculate linear velocity
        distance_to_target = math.hypot(dx, dy)
        linear_vel = min(self.kp_v * distance_to_target, self.max_linear_velocity)

        # Adjust linear velocity based on the angle difference
        if abs(alpha) > 1.0:
            linear_vel = 0.0

        # Calculate angular velocity
        angular_vel = curvature * linear_vel

        # Clip velocities
        linear_vel = max(min(linear_vel, self.max_linear_velocity), -self.max_linear_velocity)
        angular_vel = max(min(angular_vel, self.max_angular_velocity), -self.max_angular_velocity)

        return linear_vel, angular_vel

    def balance_actions(self, rl_action, pp_action):
        linear_vel = (1 - self.pp_balance_factor) * rl_action[0] + self.pp_balance_factor * pp_action[0]
        angular_vel = (1 - self.pp_balance_factor) * rl_action[1] + self.pp_balance_factor * pp_action[1]
        return [linear_vel, angular_vel]

    def execute_action(self, rl_action):
        pp_action = self.pure_pursuit_control()
        blended_action = self.balance_actions(rl_action, pp_action)
        self.send_velocity_command(blended_action)

        # Publish action type (optional)
        action_type_msg = String()
        action_type_msg.data = f"Blended (RL: {1-self.pp_balance_factor:.2f}, PP: {self.pp_balance_factor:.2f})"
        self.action_type_pub.publish(action_type_msg)

    def set_target(self, target_location):
        self._target_location = np.array(target_location, dtype=np.float32)
        self.publish_target_marker()

    def publish_target_marker(self):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.id = self.marker_id
        marker.pose.position = Point(x=float(self._target_location[0]), y=float(self._target_location[1]), z=0.0)
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = 0.2
        marker.color.a = 1.0
        marker.color.r = 1.0
        self.marker_publisher.publish(marker)
        self.get_logger().info(f"Published target marker at position: {self._target_location}")

    def call_set_robot_state_service(self, robot_pose=[-1.47, -0.045, -0.707, 0.707]):
        while not self.client_state.wait_for_service(1.0):
            self.get_logger().warn("Waiting for service...")

        request = SetEntityState.Request()
        request.state.name = self.robot_name
        request.state.pose.position.x = float(robot_pose[0])
        request.state.pose.position.y = float(robot_pose[1])
        request.state.pose.orientation.z = float(robot_pose[2])
        request.state.pose.orientation.w = float(robot_pose[3])
        request.state.twist.linear.x = request.state.twist.linear.y = request.state.twist.linear.z = 0.0
        request.state.twist.angular.x = request.state.twist.angular.y = request.state.twist.angular.z = 0.0

        future = self.client_state.call_async(request)
        future.add_done_callback(partial(self.callback_set_robot_state))

    def callback_set_robot_state(self, future):
        try:
            response = future.result()
            self.reset_path()
            self._done_set_rob_state = True
        except Exception as e:
            self.get_logger().error("Service call failed: %r" % (e,))

    def call_set_target_state_service(self, position=[-1, 4]):
        while not self.client_state.wait_for_service(1.0):
            self.get_logger().warn("Waiting for service...")

        request = SetEntityState.Request()
        request.state.name = "Target"
        request.state.pose.position.x = float(position[0])
        request.state.pose.position.y = float(position[1])

        future = self.client_state.call_async(request)
        future.add_done_callback(partial(self.callback_set_target_state))

    def callback_set_target_state(self, future):
        try:
            response = future.result()
        except Exception as e:
            self.get_logger().error("Service call failed: %r" % (e,))