#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from gymnasium.envs.registration import register
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
import os
import numpy as np
import time  

class TrainedAgent(Node):

    def __init__(self):
        super().__init__("trained_diffbot", allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)

    def stop_robot(self, env):
        """
        Function to stop the robot by sending zero velocity.
        This will stop both linear and angular motion.
        """
        try:
            # Sending zero velocity to stop the robot
            env.unwrapped.send_velocity_command([0.0, 0.0])  # Send zero velocity
            self.get_logger().info("Robot has stopped moving.")
            time.sleep(0.5)  # Allow time for the stop command to take effect
        except AttributeError:
            self.get_logger().error("Env does not have 'set_velocity' method. Modify based on your environment.")

def main(args=None):
    rclpy.init()
    node = TrainedAgent()
    node.get_logger().info("Trained agent node has been created")

    # Get the directory where the models are saved
    home_dir = os.path.expanduser('~')
    pkg_dir = '/home/vipho/rl_ws/src/rl_universe/rl_robot'
    trained_model_path = os.path.join(home_dir, pkg_dir, 'rl_models', 'SAC_waypoint03.zip')

    # Register the gym environment
    register(
        id="DiffBotEnv-v0",
        entry_point="rl_robot.diffbot_env02:DiffBotEnv",
        max_episode_steps=500,
    )

    env = gym.make('DiffBotEnv-v0')
    env = Monitor(env)
    check_env(env)

    # Load the trained model
    custom_obj = {'action_space': env.action_space, 'observation_space': env.observation_space}
    model = SAC.load(trained_model_path, env=env, custom_objects=custom_obj)

    episodes = 10

    for episode in range(episodes):
        obs, info = env.reset()  # Reset the environment
        done = False
        total_reward = 0
        step_count = 0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)  # Handle the 5-tuple return
            total_reward += reward
            step_count += 1

            # Check if the episode ends
            done = terminated or truncated

        node.get_logger().info(f"Episode {episode+1} ended with reward: {total_reward}, steps: {step_count}")

        # Stop the robot at the end of the episode
        node.stop_robot(env)
        time.sleep(5) 

        node.get_logger().info(f"Starting next episode...")

    node.get_logger().info("All episodes completed.")

    # Close the environment and destroy the node
    env.close()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
