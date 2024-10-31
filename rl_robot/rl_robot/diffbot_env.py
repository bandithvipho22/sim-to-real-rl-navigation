import rclpy
from gymnasium import Env
from gymnasium.spaces import Dict, Box
import numpy as np
from rl_robot.diffbot_controller import RobotController

# from rl_robot.real_robot_controller import RobotController

import math

class DiffBotEnv(RobotController, Env):
# class DiffBotEnv(RobotController, Env):
    def __init__(self):
        super().__init__()
        self.get_logger().info("All the publishers/subscribers have been started")

        self.robot_name = 'DiffBot'
        
        # Target locations
        
        self._target_location = np.array([0.68, 0.015,], dtype=np.float32) #1.2, 6.5

        
        self._initial_agent_location = np.array([-7.38, 1.26, 0], dtype=np.float32)
        
        # Initialize target distance parameters
        self._initial_target_distance = 1.5  # Initial distance from the robot
        self._max_target_distance = 5.0      # Maximum distance from the robot
        self._distance_increment = 1.0       # Distance increment for each new target

        self._current_target_distance = self._initial_target_distance
        self._targets_visited = 0

       
        self._randomize_env_level = 0  # Randomize only the target location
       
        self._normalize_obs = True
        self._normalize_act = True
        self._visualize_target = False #True, False
        self._reward_method = 1 #1, 2
        self._max_linear_velocity = 0.5
        self._min_linear_velocity = 0
        self._angular_velocity = 0.24
        self._minimum_dist_from_target = 0.15 #0.32, 0.5
        self._minimum_dist_from_obstacles = 0.28 #0.4
        #for huristic
        self._attraction_threshold = 3
        self._attraction_factor = 1
        self._repulsion_threshold = 1
        self._repulsion_factor = 0.1
        self._distance_penalty_factor = 1

        self._num_steps = 0
        self._num_episodes = 0
        
        # self._previous_location = np.array([0.0, 0.0])
        self._previous_orientation = np.array([0.0])
        
        # self.x_bounds = (-1.5, 1.5) 
        # self.y_bounds = (-2.5, 2.5)
        
        # self.waypoints_locations = [
        #     # First path
        #     [
        #         [-4.93, 1.34, -0.1, 0.1, -0.1, 0.2],
        #         [-4.041, -0.565, -0.1, 0.2, -0.2, 0.3],
        #         [-2.37, -0.565, -0.1, 0.1, -0.1, 0.2],
        #         [-2.37, 0.879, -0.2, 0.2, -0.08, 0.1],
        #         [-2.78, 2.668, -0.05, 0.1, -0.1, 0.1],
        #         [-1.37, 1.10, -0.2, 0.2, -0.1, 0.1],
        #         [0.299, 1.10, -0.1, 0.1, -0.2, 0.2],
        #         [2.25, 1.10, -0.1, 0.2, -0.3, 0.3],
        #         [4.43, 2.10, -0.2, 0.2, -0.1, 0.5],
        #         [7.35, 1.09, -0.1, 0.1, -0.3, 0.3]
                
        #     ],
        #     # Second path
        #     [
        #       [-5.959, 1.05, -0.2, 0.2, -0.05, 0.05],
        #       [-4.527, 0.25, -0.1, 0.1, -0.1, 0.1],
        #       [-2.53, 0.25, -0.2, 0.2, -0.3, 0.3],
        #       [-0.284, 0.788, -0.1, 0.1, -0.3, 0.3],
        #       [1.326, 0.79, -0.3, 0.3, -0.08, 0.08],
        #       [3.80, 0.022, -0.08, 0.08, -0.1, 0.1],
        #       [5.677, 1.422, -0.09, 0.09, -0.1, 0.1],
        #       [7.348, 1.42, -0.1, 0.1, -0.05, 0.05],
        #       [7.34, -0.39, -0.05, 0.05, -0.1, 0.1],
        #       [8.64, 1.052, -0.1, 0.1, -0.3, 0.3]  
        #     ],
        #     # Third path
        #     [
        #         [-6.05, 1.05, -0.1, 0.1, -0.05, 0.05],
        #         [-4.77, 1.05, -0.05, 0.05, -0.1, 0.1],
        #         [-1.70, 1.05, -0.05, 0.05, -0.01, 0.01],
        #         [-0.039, 0.237, -0.05, 0.05, -0.01, 0.01],
        #         [1.684, 1.648,  -0.05, 0.05, -0.1, 0.1],
        #         [3.32, 1.648, -0.05, 0.05, -0.1, 0.1],
        #         [5.28, 1.648, -0.08, 0.08, -0.05, 0.05],
        #         [6.67, -0.45, -0.08, 0.08, -0.05, 0.05],
        #         [7.536, 1.87, -0.08, 0.08, -0.01, 0.01],
        #         [8.61, -0.32, -0.01, 0.01, -0.01, 0.01]
        #     ],
        # ]
        
        # Improved waypoints with adjusted bounds
        self.waypoints_locations = [
             # First path
            [
                [1.2905, 0.022, -0.01, 0.01, -0.01, 0.01],  # Adjusted to be tighter initially
                [2.16, 0.0679,  -0.01, 0.01, -0.01, 0.01],
                [2.83, 0.068, -0.01, 0.01, -0.01, 0.01],
                [3.24, 0.073,  -0.01, 0.01, -0.01, 0.01],
                [3.84, 0.2405,  -0.01, 0.01, -0.01, 0.01],
                [4.33, 0.2763, -0.01, 0.01, -0.01, 0.01],
                [5.299, 0.015, -0.01, 0.01, -0.01, 0.01],
                [6.810, 0.1570,  -0.01, 0.01, -0.01, 0.01],
                [8.480, 0.1550,  -0.01, 0.01, -0.01, 0.01],
                [9.41, 0.3215,  -0.01, 0.01, -0.01, 0.01]
            ],
            # Second path
            [
                [0.605, 0.022, -0.01, 0.01, -0.01, 0.01],  # Adjusted to be tighter initially
                [1.16, 0.0679,  -0.01, 0.01, -0.01, 0.01],
                [2.83, 0.068, -0.01, 0.01, -0.01, 0.01],
                [3.24, 0.073,  -0.01, 0.01, -0.01, 0.01],
                [3.84, 0.2405,  -0.01, 0.01, -0.01, 0.01],
                [4.33, 0.2763, -0.01, 0.01, -0.01, 0.01],
                [5.299, 0.015, -0.01, 0.01, -0.01, 0.01],
                [6.210, 0.1570,  -0.01, 0.01, -0.01, 0.01],
                [7.480, 0.1550,  -0.01, 0.01, -0.01, 0.01],
                [8.41, 0.3215,  -0.01, 0.01, -0.01, 0.01]
            ],
            
        ]
        
        # self._target_location = np.array([self.target_locations[0][0], self.target_locations[0][1]], dtype=np.float32)

        self.get_logger().info("INITIAL TARGET LOCATION: " + str(self._target_location))
        self.get_logger().info("INITIAL AGENT LOCATION: " + str(self._initial_agent_location))
        self.get_logger().info("MAX LINEAR VEL: " + str(self._max_linear_velocity))
        self.get_logger().info("MIN LINEAR VEL: " + str(self._min_linear_velocity))
        self.get_logger().info("ANGULAR VEL: " + str(self._angular_velocity))
        self.get_logger().info("MIN TARGET DIST: " + str(self._minimum_dist_from_target))
        self.get_logger().info("MIN OBSTACLE DIST: " + str(self._minimum_dist_from_obstacles))

        if self._visualize_target:
            self.get_logger().info("WARNING! TARGET VISUALIZATION IS ACTIVATED, SET IT FALSE FOR TRAINING")
        # normalize action --------------------
        if self._normalize_act:
            self.action_space = Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)
        else:
            self.action_space = Box(low=np.array([self._min_linear_velocity, -self._angular_velocity]), high=np.array([self._max_linear_velocity, self._angular_velocity]), dtype=np.float32)
        # normalize observation ---------------
        if self._normalize_obs:
            self.observation_space = Dict(
                {
                    "agent": Box(low=np.array([0, 0]), high=np.array([6, 1]), dtype=np.float32),
                    "laser": Box(low=0, high=1, shape=(6,), dtype=np.float32),  # Updated shape to (10,)
                }
            )
        else:
            self.observation_space = Dict(
                {
                    "agent": Box(low=np.array([0, 0]), high=np.array([6, 1]), dtype=np.float32),
                    "laser": Box(low=0, high=1, shape=(6,), dtype=np.float32),  # Updated shape to (10,)
                }
            )

        
        # if self._normalize_obs:
        #     self.observation_space = Dict({
        #         "agent": Box(low=np.array([0, 0]), high=np.array([6, 1]), dtype=np.float32),
        #         "laser": Box(low=0, high=1, shape=(720,), dtype=np.float32),
        #     })
        # else:
        #     self.observation_space = Dict({
        #         "agent": Box(low=np.array([0, 0]), high=np.array([6, 1]), dtype=np.float32),
        #         "laser": Box(low=0, high=1, shape=(720,), dtype=np.float32),
        #     })
            
        self._successes = 0
        self._failures = 0
        self._completed_paths = 0
        
    
    def step(self, action):
        # Increase step number
        self._num_steps += 1
        
        # At the end of the step function, after updating self._agent_location
        self._previous_location = np.copy(self._agent_location)

        # De-normalize the action to send the command to robot
        if self._normalize_act:
            action = self.denormalize_action(action)

        # Apply the action
        self.send_velocity_command(action)

        # Spin the node until laser reads and agent location are updated - VERY IMPORTANT
        self.spin()

        # Compute the polar coordinates of the robot with respect to the target
        self.transform_coordinates()

        # Update robot location and laser reads
        observation = self._get_obs()
        info = self._get_info()

        # Compute Reward
        reward = self.compute_rewards(info)
        
        # Publish the marker at each step (optional)
        # self.publish_target_marker()
        
        if self._visualize_target:
            self.publish_target_marker()

        
        if self._randomize_env_level == 0:
            self.compute_statistics(info)
            
        
        # RANDOM LEVEL for (waypoint)
        if (self._which_waypoint == len(self.waypoints_locations[0])-1):
                done = (info["distance"] < self._minimum_dist_from_target) or (any(info["laser"] < self._minimum_dist_from_obstacles))
        else:
            done = (any(info["laser"] < self._minimum_dist_from_obstacles))
            # Update waypoint
            if (info["distance"] < self._minimum_dist_from_target):
                # Increase the variable to account for next waypoint
                self._which_waypoint += 1
                # Set the new waypoint
                self.randomize_target_location()
                # Here we set the new waypoint position for visualization
                if self._visualize_target == True:
                    self.call_set_target_state_service(self._target_location)

        return observation, reward, done, False, info
  

    def reset(self, seed=None, options=None):
        self._num_episodes += 1
        pose2d = self.randomize_robot_location()
        self._done_set_rob_state = False
        self.call_set_robot_state_service(pose2d)
        while self._done_set_rob_state == False:
            rclpy.spin_once(self)
            
        self._path = np.random.randint(0, len(self.waypoints_locations))
        
        self._which_waypoint = 0
        
        # Call the parent class reset method
        super().reset()

        # After resetting, publish the target marker to RViz
        self.publish_target_marker()
        
         # Initialize the agent's state
        # agent_state = np.array([0, 0], dtype=np.float32)
        
         # Retrieve lidar data (simulate or obtain from environment)
        # lidar_data = np.zeros(360, dtype=np.float32)
        
        # assert lidar_data.shape == (360,), f"Lidar data shape mismatch: expected (360,), got {lidar_data.shape}"

        # if self._randomize_env_level >= 2:
        self.randomize_target_location()

        if self._visualize_target:
            self.call_set_target_state_service(self._target_location)

        self.spin()
        self.transform_coordinates()

        observation = self._get_obs()
        info = self._get_info()
        self._num_steps = 0
        
        return observation, info

    # def _get_obs(self):
    #     # Returns the current state of the system
    #     obs = {"agent": self._polar_coordinates, "laser": self._laser_reads}
    #     # Normalize observations
    #     if self._normalize_obs == True:
    #         obs = self.normalize_observation(obs)
    #     #self.get_logger().info("Agent Location: " + str(self._agent_location))
    #     return obs
    
    # def _get_obs(self):
    #     # Returns the current state of the system
    #     obs = {"agent": self._polar_coordinates, "laser": self._laser_reads}
    #     # Normalize observations
    #     if self._normalize_obs:
    #         obs = self.normalize_observation(obs)
    #     return obs

    def _get_obs(self):
        obs = {"agent": self._polar_coordinates.astype(np.float32), "laser": self._laser_reads.astype(np.float32)}
        
        # Normalize observations
        if self._normalize_obs:
            obs = self.normalize_observation(obs)

        # Ensure the correct data types
        assert obs["laser"].dtype == np.float32, f"Laser dtype mismatch: {obs['laser'].dtype}, expected float32"
        assert obs["agent"].dtype == np.float32, f"Agent dtype mismatch: {obs['agent'].dtype}, expected float32"
        
        return obs


    def _get_info(self): #world info 
        # returns the distance from agent to target and laser reads
        return {
            "distance": math.dist(self._agent_location, self._target_location),
            "laser": self._laser_reads,
            "angle": self._theta
        }

    def spin(self):
        # This function spins the node until it gets new sensor data (executes both laser and odom callbacks)
        self._done_pose = False
        self._done_laser = False
        while (self._done_pose == False) or (self._done_laser == False):
            rclpy.spin_once(self)

    def transform_coordinates(self):
        
        self._radius = math.dist(self._agent_location[:2], self._target_location) #Euclidean distance between two points p and q.

        self._robot_target_x = math.cos(-self._agent_orientation) * (self._target_location[0] - self._agent_location[0]) - \
                               math.sin(-self._agent_orientation) * (self._target_location[1] - self._agent_location[1])
        
        self._robot_target_y = math.sin(-self._agent_orientation) * (self._target_location[0] - self._agent_location[0]) + \
                               math.cos(-self._agent_orientation) * (self._target_location[1] - self._agent_location[1])

        self._theta = math.atan2(self._robot_target_y, self._robot_target_x)
        
        self._polar_coordinates = np.array([self._radius, self._theta], dtype=np.float32)

    
    # def randomize_target_location(self):
        
    #      # The new waypoint is already set
    #     self._target_location = np.array([self.waypoints_locations[self._path][self._which_waypoint][0], self.waypoints_locations[self._path][self._which_waypoint][1]], dtype=np.float32) # Base position
    #     self._target_location[0] += np.float32(np.random.rand(1)*(self.waypoints_locations[self._path][self._which_waypoint][3]-self.waypoints_locations[self._path][self._which_waypoint][2]) + self.waypoints_locations[self._path][self._which_waypoint][2]) # Random contr. on target x
    #     self._target_location[1] += np.float32(np.random.rand(1)*(self.waypoints_locations[self._path][self._which_waypoint][5]-self.waypoints_locations[self._path][self._which_waypoint][4]) + self.waypoints_locations[self._path][self._which_waypoint][4]) # Random contr. on target y

    # def randomize_target_location(self):
    #     # Get the current waypoint
    #     current_waypoint = self.waypoints_locations[self._path][self._which_waypoint]
        
    #     # Extract base position and randomization ranges
    #     base_x, base_y = current_waypoint[:2]
    #     x_range = current_waypoint[2:4]
    #     y_range = current_waypoint[4:6]
        
    #     # Generate random offsets within the specified ranges
    #     x_offset = np.random.uniform(*x_range)
    #     y_offset = np.random.uniform(*y_range)
        
    #     # Set the new target location
    #     self._target_location = np.array([
    #         base_x + x_offset,
    #         base_y + y_offset
    #     ], dtype=np.float32)
        
    #     # Optional: Add some noise to prevent overfitting
    #     noise = np.random.normal(0, 0.01, 2)  # Small Gaussian noise
    #     self._target_location += noise
        
    #     # # Optional: Ensure the target is within the environment bounds
    #     # self._target_location = np.clip(self._target_location, 
    #     #                                 self.env_bounds[:, 0], 
    #     #                                 self.env_bounds[:, 1])

    #     return self._target_location
    
    def randomize_target_location(self):
        current_waypoint = self.waypoints_locations[self._path][self._which_waypoint]
        
        base_x, base_y = current_waypoint[:2]
        x_range = current_waypoint[2:4]
        y_range = current_waypoint[4:6]
        
        x_offset = np.random.uniform(*x_range)
        y_offset = np.random.uniform(*y_range)
        
        self._target_location = np.array([
            base_x + x_offset,
            base_y + y_offset
        ], dtype=np.float32)
        
        # Optional: Add some noise to prevent overfitting
        noise = np.random.normal(0, 0.01, 2)
        self._target_location += noise
        
        # # Optional: Basic sanity check for bounds
        # if hasattr(self, 'x_bounds') and hasattr(self, 'y_bounds'):
        #     self._target_location[0] = np.clip(self._target_location[0], self.x_bounds[0], self.x_bounds[1])
        #     self._target_location[1] = np.clip(self._target_location[1], self.y_bounds[0], self.y_bounds[1])

        return self._target_location
    
    def randomize_robot_location(self):
        
        
        position_x = float(self._initial_agent_location[0])
        position_y = float(self._initial_agent_location[1])
        angle = float(math.radians(self._initial_agent_location[2]))
        orientation_z = float(math.sin(angle / 2))
        orientation_w = float(math.cos(angle / 2))
    
            
        return [position_x, position_y, orientation_z, orientation_w]
    
    
    def compute_rewards(self, info):
        # This method computes the reward of the step

        # Risk seeker reward
        if self._reward_method == 1:
            if (info["distance"] < self._minimum_dist_from_target):
                # If the agent reached the target it gets a positive reward
                reward = 10
                self.get_logger().info("TARGET REACHED")
                self.get_logger().info("Agent: X = " + str(self._agent_location[0]) + " - Y = " + str(self._agent_location[1]))
            
            elif (any(info["laser"] < self._minimum_dist_from_obstacles)):
                # If the agent hits an abstacle it gets a negative reward
                reward = -1
                self.get_logger().info("HIT AN OBSTACLE")
            else:
                # Otherwise the episode continues
                reward = 0

        # Heuristic with Adaptive Exploration Strategy
        if self._reward_method == 2:
            if info["distance"] < self._minimum_dist_from_target:
                # If the agent reached the target, it gets a positive reward minus the time it took
                reward = 1000 - self._num_steps
                self.get_logger().info("TARGET REACHED")
                self.get_logger().info("Agent: X = " + str(self._agent_location[0]) + " - Y = " + str(self._agent_location[1]))

            elif any(info["laser"] < self._minimum_dist_from_obstacles):
                # If the agent hits an obstacle, it gets a negative reward
                reward = -500 #-10000
                self.get_logger().info("HIT AN OBSTACLE")

            else:
                # Instant reward of the current state - based on Euclidean distance to the target
                instant_reward = -info["distance"] * self._distance_penalty_factor

                # Estimated reward of the current state (attraction-repulsion rule)
                # Attraction factor - Activates when the agent is near the target
                attraction_reward = self._attraction_factor / info["distance"] if info["distance"] < self._attraction_threshold else 0

                # Repulsion factor - Activates when the agent is near any obstacles
                contributions = [((-self._repulsion_factor / read**2) * ((1/read) - (1/self._repulsion_threshold))) 
                                for read in info["laser"] if read <= self._repulsion_threshold]
                repulsion_reward = sum(contributions)

                # Compute the final reward, capping the max reward at 1
                reward = min(instant_reward + attraction_reward + repulsion_reward, 1)

            
        return reward

    
    # def normalize_observation(self, observation):
    #     ## This method normalizes the observations taken from the robot in the range [0,1]
    #     # Distance from target can range from 0 to 60 but we divide by ten since most times the agent never goes further
    #     observation["agent"][0] = observation["agent"][0]/10
    #     # Angle from target can range from -pi to pi
    #     observation["agent"][1] = (observation["agent"][1] + math.pi)/(2*math.pi)
    #     # Laser reads range from 0 to 10
    #     observation["laser"] = observation["laser"]/10

        
    #     return observation
    
    def normalize_observation(self, observation):
        # Normalize distance (assuming max distance is 10)
        observation["agent"][0] = observation["agent"][0] / 12.0
        
        # Normalize angle: Convert the angle from the range [-pi, pi] to [0, 1]
        observation["agent"][1] = (observation["agent"][1] + math.pi) / (2 * math.pi)
        
        # Normalize laser readings
        observation["laser"] = observation["laser"] / 12.0

        return observation



    def denormalize_action(self, norm_act):
        ## This method de-normalizes the action before sending it to the robot - The action is normalized between [-1,1]
        # Linear velocity can also be asymmetric
        action_linear = ((self._max_linear_velocity*(norm_act[0]+1)) + (self._min_linear_velocity*(1-norm_act[0])))/2
        # Angular velicity is symmetric
        action_angular = ((self._angular_velocity*(norm_act[1]+1)) + (-self._angular_velocity*(1-norm_act[1])))/2

      
        return np.array([action_linear, action_angular], dtype=np.float32)

    def compute_statistics(self, info):
        ## This method is used to compute statistic
        if (info["distance"] < self._minimum_dist_from_target):
                # If the agent reached the target it gets a positive reward
                self._successes += 1
            
        elif (any(info["laser"] < self._minimum_dist_from_obstacles)):
                # If the agent hits an abstacle it gets a negative reward
                self._failures += 1
        else:
            pass

    def close(self):
      
        if self._randomize_env_level == 0:
            self.get_logger().info("Completed paths: " + str(self._completed_paths))
            self.get_logger().info("Truncated episodes: " + str(self._num_episodes-1 - self._completed_paths - self._failures))
            self.get_logger().info("Avg. targets reached x episode: " + str(self._successes/(self._num_episodes-1)))
        else:
            self.get_logger().info("Truncated episodes: " + str(self._num_episodes-1 - self._successes - self._failures))

        # Destroy all clients/publishers/subscribers
        """self.destroy_client(self.client_sim)
        self.destroy_client(self.client_state)
        self.destroy_publisher(self.action_pub)
        self.destroy_subscription(self.pose_sub)
        self.destroy_subscription(self.laser_sub)"""
        self.destroy_node()