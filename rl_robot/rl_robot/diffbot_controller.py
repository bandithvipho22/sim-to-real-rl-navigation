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
import os
from ament_index_python.packages import get_package_share_directory
#from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from std_msgs.msg import String

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point 

class RobotController(Node):
    """
    This class defines all the methods to:
        - Publish actions to the agent (move the robot)
        - Subscribe to sensors of the agent (get laser scans and robot position)
        - Reset the simulation

    Topics list:
        - /demo/cmd_vel : linear and angular velocity of the robot
        - /demo/odom : odometry readings of the chassis of the robot
        - /scan: laser readings
    
    Services used:
        - /demo/set_entity_state : sets the new state of the robot and target when an episode ends

    """
    def __init__(self):
        
        super().__init__('robot_controller')
        self.get_logger().info("The robot controller node has just been created")

        # Action publisher
        self.action_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        # self.action_sub = self.create_subscription(Twist, '/demo/cmd_vel', self.cmd_callback, 1)
        # Position subscriber
        self.pose_sub = self.create_subscription(Odometry, '/demo/odom', self.pose_callback, 10)
        # Laser subscriber
        self.laser_sub = self.create_subscription(LaserScan, '/scan', self.laser_callback, 100)
        # Reset model state client - this resets the pose and velocity of a given model within the world
        self.client_state = self.create_client(SetEntityState, "/demo/set_entity_state")
        
        self.path_pub = self.create_publisher(Path, '/robot_path', 10)
        
        # self.action_type_pub = self.create_publisher(String, '/agent_action_type', 10)
        self.action_type_pub = self.create_publisher(String, '/agent_action_type', 10)

        # Initialize attributes - This will be immediately re-written when the simulation starts
        self._agent_location = np.array([np.float32(1),np.float32(16)])
        # self._laser_reads = np.array([np.float32(10)] * 61)
        # If downsampling to 10 points
        # Initialize for 60 points
        self._laser_reads = np.array([np.float32(10)] * 60)
 

        # Initialize path message
        self.path = Path()  
        self.path.header.frame_id = "odom"
        
        # ROS2 marker publisher for RViz
        self.marker_publisher = self.create_publisher(Marker, '/marker', 10)
        self.marker_id = 0
        
    def publish_target_marker(self):
        """
        Publish a marker to RViz to visualize the current target location.
        """
        marker = Marker()
        marker.header.frame_id = "map"  
        marker.header.stamp = self.get_clock().now().to_msg()

        # Set the marker's type
        marker.type = Marker.SPHERE  # Visualize the target as a sphere
        marker.action = Marker.ADD
        marker.id = self.marker_id

        # Define the position of the marker (the target position)
        marker.pose.position = Point(x=float(self._target_location[0]), y=float(self._target_location[1]), z=0.0)
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        # Define the scale of the marker (size of the sphere)
        marker.scale.x = 0.2 
        marker.scale.y = 0.2  
        marker.scale.z = 0.2  

        # Define the color of the marker
        marker.color.a = 1.0  
        marker.color.r = 1.0
        marker.color.g = 0.0  
        marker.color.b = 0.0

        # Publish the marker
        self.marker_publisher.publish(marker)
        self.get_logger().info(f"Published target marker at position: {self._target_location}")
    
    
    # Method to send the velocity command to the robot
    def send_velocity_command(self, velocity):
        msg = Twist()
        msg.linear.x = float(velocity[0])
        msg.angular.z = float(velocity[1])
        self.action_pub.publish(msg)
        
    #     self.action_type = self.determine_action_type(velocity)
        
    #     # Publish the action type
    #     self.action_type_pub(self.action_type)
        
    # def determine_action_type(self, action):
    #     # Determine the agent's action direction
    #     linear_velocity = action[0]
    #     angular_velocity = action[1]

    #     if angular_velocity > 0.2:
    #         self.action_type = "Turning Right"
    #     elif angular_velocity < -0.2:
    #        self.action_type = "Turning Left"
    #     else:
    #         self.action_type = "Going Forward"

        # self.get_logger().info(f"Action Taken: {action_type}, Linear Velocity: {linear_velocity}, Angular Velocity: {angular_velocity}")

    # Method that saves the position of the robot each time the topic /demo/odom receives a new message
    def pose_callback(self, msg: Odometry):
        self._agent_location = np.array([np.float32(np.clip(msg.pose.pose.position.x,-12,12)), np.float32(np.clip(msg.pose.pose.position.y,-35,21))])
        self._agent_orientation = 2* math.atan2(msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)
     
        self._done_pose = True
        
         # Update and publish path
        self.update_and_publish_path(msg)
        
    # Method to update and publish the robot's path
    def update_and_publish_path(self, msg: Odometry):
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()  # Use current time
        pose.header.frame_id = "odom"
        pose.pose = msg.pose.pose
        self.path.poses.append(pose)  # Add new pose to the path
        self.path_pub.publish(self.path)  # Publish the path
        
    # Method to reset the path
    def reset_path(self):
        self.path.poses = [] 
        
    def laser_callback(self, msg: LaserScan):
        self._done_laser = False  # Set the flag to indicate processing
        
        # Convert the laser data to a numpy array
        full_laser_reads = np.array(msg.ranges)
        
        # Handle infinite values by replacing them with a large value (e.g., 10.0)
        full_laser_reads[full_laser_reads == np.inf] = np.float32(10)
        
        # We only want the front 180 degrees (assuming full_laser_reads is 360 degrees)
        num_laser_points = len(full_laser_reads)
        front_laser_reads = full_laser_reads[:num_laser_points // 2]  
        
        # Downsample the front laser data to 6 points using interpolation
        downsampled_laser_reads = np.interp(
            np.linspace(0, len(front_laser_reads) - 1, 6),  #Points at which to sample, downsampled to 6 points
            np.arange(len(front_laser_reads)),  # Original indices of the front 180-degree data
            front_laser_reads  # Front 180-degree laser data
        )

        # Ensure there are no NaN values in the data
        self._laser_reads = np.nan_to_num(downsampled_laser_reads, nan=np.float32(10))

        # Set the flag to indicate the laser data has been processed
        self._done_laser = True


    # # Method that saves the laser reads each time the topic /demo/laser/out receives a new message
    # def laser_callback(self, msg: LaserScan):
    #     self._laser_reads = np.array(msg.ranges)
    #     # Converts inf values to 10
    #     self._laser_reads[self._laser_reads == np.inf] = np.float32(10)
    #     #self.get_logger().info("Min Laser Read: " + str(min(self._laser_reads)))
    #     self._done_laser = True
    
    # def laser_callback(self, msg: LaserScan):
    #     """
    #     Callback for laser scan data. Downsamples the laser scan to match the expected shape.
    #     """
    #     if msg.ranges:
    #         full_laser_reads = np.array(msg.ranges)

    #         # Ensure that the full laser reads are downsampled to 10 points
    #         num_laser_points = 10
    #         downsampled_laser_reads = np.interp(
    #             np.linspace(0, len(full_laser_reads) - 1, num_laser_points),
    #             np.arange(len(full_laser_reads)),
    #             full_laser_reads
    #         )

    #         # Replace inf with a maximum range value (e.g., 10.0)
    #         downsampled_laser_reads[downsampled_laser_reads == np.inf] = 10.0
    #         # Replace NaNs with a default value (e.g., 10.0)
    #         self._laser_reads = np.nan_to_num(downsampled_laser_reads, nan=10.0)
    #     else:
    #         self.get_logger().error("Invalid laser scan data received!")


    
    # def laser_callback(self, msg: LaserScan):
    #     """
    #     Callback function for laser scan data. Handle infinite ranges and NaN values safely.
    #     """
    #     if msg.ranges:
    #         self._laser_reads_full = np.array(msg.ranges)
    #         self._laser_reads_full[self._laser_reads_full == np.inf] = 10.0  # Replace inf with max range
    #         self._laser_reads_full = np.nan_to_num(self._laser_reads_full, nan=10.0)  # Replace NaNs with max range
            
    #         # Downsample the 720 laser readings to 6 values (e.g., by averaging regions)
    #         self._laser_reads = np.array([
    #             np.mean(self._laser_reads_full[0:120]),    # Front left
    #             np.mean(self._laser_reads_full[120:240]),  # Left
    #             np.mean(self._laser_reads_full[240:360]),  # Back left
    #             np.mean(self._laser_reads_full[360:480]),  # Back right
    #             np.mean(self._laser_reads_full[480:600]),  # Right
    #             np.mean(self._laser_reads_full[600:720])   # Front right
    #         ], dtype=np.float32)
    #     else:
    #         self.get_logger().error("Invalid laser scan data received!")


    # Method to set the state of the robot when an episode ends - /demo/set_entity_state service
    def call_set_robot_state_service(self, robot_pose=[-1.47, -0.045, -0.707, 0.707]):
    
    # def call_set_robot_state_service(self, robot_pose=[0.0, 0.0, 0.0, 0.0]):
    
        while not self.client_state.wait_for_service(1.0):
            self.get_logger().warn("Waiting for service...")

        request = SetEntityState.Request()
        request.state.name = self.robot_name
        # Pose (position and orientation)
        request.state.pose.position.x = float(robot_pose[0])
        request.state.pose.position.y = float(robot_pose[1])
        request.state.pose.orientation.z = float(robot_pose[2])
        request.state.pose.orientation.w = float(robot_pose[3])
        # Velocity
        request.state.twist.linear.x = float(0)
        request.state.twist.linear.y = float(0)
        request.state.twist.linear.z = float(0)
        request.state.twist.angular.x = float(0)
        request.state.twist.angular.y = float(0)
        request.state.twist.angular.z = float(0)

        future = self.client_state.call_async(request)
        future.add_done_callback(partial(self.callback_set_robot_state))

    # Method that elaborates the future obtained by callig the call_set_robot_state_service method
    def callback_set_robot_state(self, future):
        try:
            response = future.result()
            # Reset path after resetting robot state
            self.reset_path()
            #self.get_logger().info("The Environment has been successfully reset")
            self._done_set_rob_state = True
        except Exception as e:
            self.get_logger().error("Service call failed: %r" % (e,))

    # Method to set the state of the target when an episode ends - /demo/set_entity_state service
    def call_set_target_state_service(self, position=[-1, 4]): #[1, 10]
        while not self.client_state.wait_for_service(1.0):
            self.get_logger().warn("Waiting for service...")

        request = SetEntityState.Request()
        request.state.name = "Target"
        # Pose (position and orientation)
        request.state.pose.position.x = float(position[0])
        request.state.pose.position.y = float(position[1])

        future = self.client_state.call_async(request)
        future.add_done_callback(partial(self.callback_set_target_state))

    # Method that elaborates the future obtained by callig the call_set_target_state_service method
    def callback_set_target_state(self, future):
        try:
            response= future.result()
            #self.get_logger().info("The Environment has been successfully reset")
        except Exception as e:
            self.get_logger().error("Service call failed: %r" % (e,))

   
# from rclpy.node import Node
# from sensor_msgs.msg import LaserScan
# from nav_msgs.msg import Odometry
# from geometry_msgs.msg import Twist, PoseStamped, Point
# from nav_msgs.msg import Path
# from std_msgs.msg import String
# from visualization_msgs.msg import Marker
# import numpy as np
# import math

# class RobotController(Node):
#     def __init__(self):
#         super().__init__('real_robot_controller')
#         self.get_logger().info("Real robot controller node has been created")

#         # Action publisher
#         self.action_pub = self.create_publisher(Twist, '/cmd_vel', 10)

#         # Position subscriber
#         self.pose_sub = self.create_subscription(Odometry, '/odom', self.pose_callback, 1)

#         # Laser subscriber
#         self.laser_sub = self.create_subscription(LaserScan, '/scan', self.laser_callback, 1)

#         # Path publisher for visualization
#         self.path_pub = self.create_publisher(Path, '/robot_path', 10)

#         # Action type publisher
#         self.action_type_pub = self.create_publisher(String, '/agent_action_type', 10)

#         # Marker publisher for target visualization
#         self.marker_publisher = self.create_publisher(Marker, '/target_marker', 10)

#         # Initialize attributes
#         self._agent_location = np.array([np.float32(0), np.float32(0)])
#         self._laser_reads = np.array([np.float32(10)] * 61)

#         # Path message for visualization
#         self.path = Path()  
#         self.path.header.frame_id = "odom"
#         self.marker_id = 0

#     def publish_target_marker(self, target_location):
#         """
#         Publish a marker to RViz to visualize the current target location.
#         """
#         marker = Marker()
#         marker.header.frame_id = "map"
#         marker.header.stamp = self.get_clock().now().to_msg()

#         marker.type = Marker.SPHERE
#         marker.action = Marker.ADD
#         marker.id = self.marker_id

#         # Set marker's position to the target's location
#         marker.pose.position = Point(x=float(target_location[0]), y=float(target_location[1]), z=0.0)
#         marker.scale.x = 0.2
#         marker.scale.y = 0.2
#         marker.scale.z = 0.2

#         marker.color.a = 1.0
#         marker.color.r = 1.0
#         marker.color.g = 0.0
#         marker.color.b = 0.0

#         self.marker_publisher.publish(marker)
#         self.get_logger().info(f"Published target marker at position: {target_location}")

#     def send_velocity_command(self, velocity):
#         """
#         Send a velocity command to the robot.
#         """
#         msg = Twist()
#         msg.linear.x = float(velocity[0])
#         msg.angular.z = float(velocity[1])
#         self.action_pub.publish(msg)

#     def pose_callback(self, msg: Odometry):
#         """
#         Callback to update the robot's current location and orientation.
#         """
#         self._agent_location = np.array([
#             np.float32(msg.pose.pose.position.x),
#             np.float32(msg.pose.pose.position.y)
#         ])
#         self._agent_orientation = 2 * math.atan2(msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)

#         self.get_logger().info(f"Robot position: {self._agent_location}, orientation: {math.degrees(self._agent_orientation)}")

#         # Update and publish the robot's path
#         self.update_and_publish_path(msg)

#     def update_and_publish_path(self, msg: Odometry):
#         """
#         Update and publish the robot's path for visualization.
#         """
#         pose = PoseStamped()
#         pose.header.stamp = self.get_clock().now().to_msg()
#         pose.header.frame_id = "odom"
#         pose.pose = msg.pose.pose
#         self.path.poses.append(pose)
#         self.path_pub.publish(self.path)

#     def reset_path(self):
#         """
#         Reset the robot's path.
#         """
#         self.path.poses = []

#     def laser_callback(self, msg: LaserScan):
#         """
#         Callback to update laser readings.
#         """
#         self._laser_reads = np.array(msg.ranges)
#         self._laser_reads[self._laser_reads == np.inf] = np.float32(10)
#         self.get_logger().info(f"Laser min read: {min(self._laser_reads)}")

#     def move_to_target(self, target_location):
#         """
#         Implement your control logic to move the robot to the target.
#         This can be done using motion planning or PID control, depending on your setup.
#         """
#         # Example logic (add your own implementation):
#         while not self.reached_target(target_location):
#             velocity = self.compute_velocity_to_target(target_location)
#             self.send_velocity_command(velocity)
#         self.get_logger().info(f"Reached target at {target_location}")

#     def compute_velocity_to_target(self, target_location):
#         """
#         Compute the linear and angular velocity to move towards the target.
#         """
#         # Simple proportional control (example, replace with your own control logic):
#         delta_x = target_location[0] - self._agent_location[0]
#         delta_y = target_location[1] - self._agent_location[1]
#         distance_to_target = math.sqrt(delta_x**2 + delta_y**2)
#         angle_to_target = math.atan2(delta_y, delta_x)

#         # Proportional control for linear and angular velocity
#         linear_velocity = 0.5 * distance_to_target
#         angular_velocity = 2.0 * (angle_to_target - self._agent_orientation)

#         return [linear_velocity, angular_velocity]

#     def reached_target(self, target_location):
#         """
#         Check if the robot has reached the target.
#         """
#         distance_to_target = np.linalg.norm(target_location - self._agent_location)
#         return distance_to_target < 0.1
