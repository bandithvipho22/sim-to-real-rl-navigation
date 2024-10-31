#!usr/bin/env python3

import rclpy
from rclpy.node import Node
from gymnasium.envs.registration import register
# from hospital_robot_spawner.hospitalbot_env import DiffBotEnv
# from hospital_robot_spawner.hospitalbot_simplified_env import HospitalBotSimpleEnv
import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import os
import optuna
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

class TrainingNode(Node):

    def __init__(self):
        super().__init__("diffbot_training", allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)

        self._training_mode = "training" #"random_agent" #"training"


def main(args=None):

    # Initialize the training node to get the desired parameters
    rclpy.init()
    node = TrainingNode()
    node.get_logger().info("Training node has been created")

    # Create the dir where the trained RL models will be saved
    home_dir = os.path.expanduser('~')
    pkg_dir = os.path.join(os.getcwd(), 'src/rl_universe/rl_robot')
    trained_models_dir = os.path.join(home_dir, pkg_dir, 'rl_models')
    log_dir = os.path.join(home_dir, pkg_dir, 'logs')
    
    # If the directories do not exist we create them
    if not os.path.exists(trained_models_dir):
        os.makedirs(trained_models_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # First we register the gym environment created in hospitalbot_env module
    register(
        id="DiffBotEnv-v0",
        entry_point="rl_robot.diffbot_env:DiffBotEnv",
        max_episode_steps= 300 #300,
    )

    node.get_logger().info("The environment has been registered")

    #env = NormalizeReward(gym.make('DiffBotEnv-v0'))
    env = gym.make('DiffBotEnv-v0')
    env = Monitor(env)

    # Sample Observation and Action space for Debugging
    #node.get_logger().info("Observ sample: " + str(env.observation_space.sample()))
    #node.get_logger().info("Action sample: " + str(env.action_space.sample()))

    # Here we check if the custom gym environment is fine
    check_env(env)
    node.get_logger().info("Environment check finished")

    # Now we create two callbacks which will be executed during training
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=900, verbose=1)
    eval_callback = EvalCallback(env, callback_on_new_best=stop_callback, eval_freq=100000, best_model_save_path=trained_models_dir, n_eval_episodes=40)
    
    if node._training_mode == "random_agent":
        # N° Episodes
        episodes = 8
        ## Execute a random agent
        node.get_logger().info("Starting the RANDOM AGENT now")
        for ep in range(episodes):
            obs = env.reset()
            done = False
            while not done:
                obs, reward, done, truncated, info = env.step(env.action_space.sample())
                node.get_logger().info("Agent state: [" + str(info["distance"]) + ", " + str(info["angle"]) + "]")
                node.get_logger().info("Reward at step " + ": " + str(reward))
    
    elif node._training_mode == "training":
        ## Train the model
        #model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=log_dir, n_steps=2279, gamma=0.9880614935504514, gae_lambda=0.9435887928788405, ent_coef=0.00009689939917928778, vf_coef=0.6330533453055319, learning_rate=0.00011770118633714448, clip_range=0.1482)
        
        # model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=log_dir, n_steps=20480, gamma=0.9880614935504514, gae_lambda=0.9435887928788405, ent_coef=0.00009689939917928778, vf_coef=0.6330533453055319, learning_rate=0.00001177011863371444, clip_range=0.1482)
        
        
        model = SAC(
            "MultiInputPolicy", 
            env, 
            verbose=1, 
            tensorboard_log=log_dir, 
            learning_rate=0.0003, #0.00005
            buffer_size=1000000, 
            learning_starts=100, 
            batch_size=256,  #64
            tau=0.005, 
            gamma=0.99, 
            train_freq=10, 
            gradient_steps=1,  #default 1
            ent_coef=  'auto', #0.01,  # Set to a small value like 0.01
            target_update_interval=1, 
            target_entropy='auto',  # Use 'auto' to let the algorithm decide the target entropy
            use_sde=False, 
            sde_sample_freq=-1, 
            use_sde_at_warmup=False, 
            stats_window_size=100, 
            seed=None, 
            device='auto'
        )

        # Execute training
        try:
            model.learn(total_timesteps=int(4000000), reset_num_timesteps=False, callback=eval_callback, tb_log_name="SAC_test")
        except KeyboardInterrupt:
            model.save(f"{trained_models_dir}/SAC_test")
        # Save the trained model
        model.save(f"{trained_models_dir}/SAC_test")
    
    elif node._training_mode == "retraining":
        ## Re-train an existent model
        node.get_logger().info("Retraining an existent model")
        # Path in which we find the model
        trained_model_path = os.path.join(home_dir, pkg_dir, 'rl_models', 'SAC_waypoint01.zip')
        # Here we load the rained model
        custom_obj = {'action_space': env.action_space, 'observation_space': env.observation_space}
        # model = PPO.load(trained_model_path, env=env, custom_objects=custom_obj)
        model = SAC.load(trained_model_path, env=env, custom_objects=custom_obj)
        
        # Execute training
        try:
            model.learn(total_timesteps=int(40000000), reset_num_timesteps=False, callback=eval_callback, tb_log_name="SAC_waypoint01")
        except KeyboardInterrupt:
            # If you notice that the training is sufficiently well interrupt to save
            model.save(f"{trained_models_dir}/SAC_waypoint01")
        # Save the trained model
        model.save(f"{trained_models_dir}/SAC_waypoint01")

   
    # Shutting down the node
    node.get_logger().info("The training is finished, now the node is destroyed")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()