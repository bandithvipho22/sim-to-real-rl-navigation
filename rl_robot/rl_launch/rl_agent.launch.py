import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    ld = LaunchDescription()

    trained_agent = Node(
        package='rl_robot',
        executable='rl_agent02',
      
    )

    ld.add_action(trained_agent)

    return ld