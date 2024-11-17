# Sim-To-Real Reinforcement Learning for Robot navigation
Zero-shot sim-to-real learning, instead, enables transferring the model from simulation to reality without using the data of the reality. Domain randomization [Reference Tobin, Fong, Ray, Schneider, Zaremba and Abbeel4, Reference Peng, Andrychowicz, Zaremba and Abbeel5] randomizes the visual information or dynamic parameters in the simulated environment, which is a popular method in zero-shot sim-to-real transfer. We also use the domain randomization method to train the model in the simulator.
## Project Overview
This project inspire from the project base [“Tommaso Van Der Meer”](https://github.com/TommasoVandermeer/Hospitalbot-Path-Planning) which is using
reinforcement learning for robot navigation in “Hospital world” using ROS2, and Gazebo
frame work for simulation.
## Objective
The primary of this project is using Sim-To-Real Reinforcement Learning for robot self-
navigation from current position to a predefined target point with unknown Environment. We
train the rl_model in ROS2 Gazebo framework and directly deploy in Algobot by using zero-shot
sim-to-real without fine-tuning.
## Output Features
+ Setup World in GAZEBO with Dimension (4x15m)
+ Visualize and tracking path robot through RVIZ
+ Using SAC algorithm to train RL model
+ Train RL model by following Path which construct with 10 target points
+ RL model "SAC_waypoint03.zip" transfer model to Real Robot
## Software Requirement
+ Ubuntu 22.04
+ ROS2 Humble
+ Gazebo Framework
+ gym library
+ Stable baseline framework
+ pytorch
+ Micro-ROS (Setup in this [LINK](https://github.com/micro-ROS/micro_ros_arduino/tree/humble)
## Scope of works
+ Create way-points for Path Planing on mobile robot navigation
+ Configure robot’s parameters for model training according to Algobot
+ Training RL Model in Simulation (Gazebo World)
+ Hardware Implementation -> Deploy RL model into Algobot

![image](https://github.com/user-attachments/assets/1004fe45-887f-4d8c-83c8-ac4a13c2ae5c)

# Simulation
## Start Training
First, create workspace
``` bash
mkdir -p rl_ws/src
cd rl_ws/src
git clone https://github.com/bandithvipho22/sim-to-real-rl-navigation.git
```
Then, go to directory and build
``` bash
cd rl_ws/
colcon build
```
Run GAZEBO to check it work or not
``` bash
ros2 launch rl_gazebo gazebo.launch.py
```
or, you can use this while training
``` bash
ros2 launch rl_gazebo headless_world.launch.py
```
And, run this to start training rl model
``` bash
ros2 launch rl_robot rl_training.launch.py
```
You can visualize how the robot work by visual in RVIZ2
``` bash
ros2 launch rl_rviz rviz.launch.py
```
## Run Model
After, finished training we can use this command below to evaluate and see the performance of robot
``` bash
ros2 launch rl_robot rl_agent.launch.py
```
You can adjust path and total episode in file "diffbot_env.py and rl_agent.py"
