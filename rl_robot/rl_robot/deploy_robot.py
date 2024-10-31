import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from stable_baselines3 import SAC
import numpy as np
import math
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker

class RLRobotController(Node):
    def __init__(self):
        super().__init__('rl_robot_controller')

        # Load the trained RL model (SAC)
        home_dir = os.path.expanduser('~')
        pkg_dir = '/home/vipho/rl_ws/src/rl_universe/rl_robot'
        trained_model_path = os.path.join(home_dir, pkg_dir, 'rl_models', 'SAC_waypoint01.zip')
        self.model = SAC.load(trained_model_path)

        self.get_logger().info(f"Loaded SAC model from {trained_model_path}")

        # ROS2 publishers and subscribers
        self.action_pub = self.create_publisher(Twist, '/cmd_vel', 10)  # Publish robot actions (velocity)
        self.pose_sub = self.create_subscription(Odometry, '/demo/odom', self.pose_callback, 10)  # Subscribe to odometry
        self.laser_sub = self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)  # Subscribe to laser scan

        # ROS2 Marker Publisher for RViz visualization
        self.marker_pub = self.create_publisher(Marker, '/marker', 10)

        # Initialize robot's state
        self._agent_location = np.array([0.0, 0.0], dtype=np.float32)
        self._agent_orientation = 0.0  # Agent orientation (theta)
        self._laser_reads = np.array([10.0] * 61, dtype=np.float32)  # Default laser scan data

        # Define waypoints for navigation
        self.waypoints = [
            [-5.8, 1.15],  # Waypoint 1
            [-4.93, 1.34],  # Waypoint 2
            [-3.73, 1.15],  # Waypoint 3
        ]
        self.current_waypoint_idx = 0  # Start at the first waypoint
        self.target_location = np.array(self.waypoints[self.current_waypoint_idx], dtype=np.float32)

        # Set goal parameters
        self._minimum_dist_from_target = 0.5
        self._minimum_dist_from_obstacles = 0.4

   
    def check_reached_waypoint(self):
        """Check if the robot reached the current waypoint."""
        distance_to_waypoint = np.linalg.norm(self._agent_location - self.target_location)

        if distance_to_waypoint < self._minimum_dist_from_target:
            self.get_logger().info(f"Reached waypoint {self.current_waypoint_idx + 1}")

            # Move to the next waypoint
            self.current_waypoint_idx = (self.current_waypoint_idx + 1) % len(self.waypoints)
            self.target_location = np.array(self.waypoints[self.current_waypoint_idx], dtype=np.float32)
            self.get_logger().info(f"Next waypoint: {self.target_location}")
            self.publish_target_marker(self.target_location)

    def publish_target_marker(self, location):
        """Publish a marker to RViz for the current target location."""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()

        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position = Point(x=float(location[0]), y=float(location[1]), z=0.0)
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0

        self.marker_pub.publish(marker)

    def pose_callback(self, msg):
        """Callback for odometry data to update robot position and orientation."""
        self._agent_location = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y], dtype=np.float32)
        self._agent_orientation = 2 * math.atan2(msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)

        # Log robot position and orientation
        self.get_logger().info(f"Robot position: {self._agent_location}, orientation: {self._agent_orientation}")

        # Check if robot reached the current waypoint
        self.check_reached_waypoint()

    def laser_callback(self, msg):
        """Callback for laser scan data."""
        self._laser_reads = np.array(msg.ranges)
        self._laser_reads[self._laser_reads == np.inf] = 10.0  # Replace infinite readings with max range
        self._laser_reads = np.nan_to_num(self._laser_reads, nan=10.0)  # Replace NaNs with max range

        # Log laser scan data
        self.get_logger().info(f"Received laser scan: {self._laser_reads[:6]}")  # Log first 6 values for brevity

    def send_action(self):
        """Generate an action using the trained RL model and send it to the robot."""
        # Generate observation
        observation = self.get_observation()

        # Log observation
        self.get_logger().info(f"Observation: {observation}")

        # Generate action using the SAC model
        try:
            action, _states = self.model.predict(observation)
            self.get_logger().info(f"Generated action: {action}")
        except Exception as e:
            self.get_logger().error(f"Error generating action: {e}")
            return

        # Send the predicted action to the robot
        self.send_velocity_command(action)


    def get_observation(self):
        """Construct the observation for the RL agent."""
        # Observation is based on the robot's location, orientation, and laser scan
        distance_to_target = np.linalg.norm(self._agent_location - self.target_location)
        angle_to_target = math.atan2(
            self.target_location[1] - self._agent_location[1],
            self.target_location[0] - self._agent_location[0]
        ) - self._agent_orientation

        # Wrap angle within [-pi, pi]
        angle_to_target = (angle_to_target + np.pi) % (2 * np.pi) - np.pi

        # Downsample the laser scan to 6 values (using slicing)
        # Select 6 equally spaced laser readings from the 61 values
        downsampled_laser = self._laser_reads[::10][:6]  # Take every 10th value from the laser readings

        observation = {
            "agent": np.array([distance_to_target, angle_to_target]),
            "laser": downsampled_laser  # Use the downsampled laser data
        }

        return observation


    def send_velocity_command(self, action):
        """Send the velocity command to the robot."""
        msg = Twist()
        msg.linear.x = float(action[0])  # Linear velocity
        msg.angular.z = float(action[1])  # Angular velocity
        self.action_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = RLRobotController()

    try:
        # Loop rate and action execution
        rate = node.create_rate(10)  # 10 Hz loop rate
        while rclpy.ok():
            rclpy.spin_once(node)
            node.send_action()
            rate.sleep()
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down RL robot controller.')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
