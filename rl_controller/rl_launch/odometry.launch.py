from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='rl_controller',
            executable='imu_pub',
            output='screen',
        ),
        Node(
            package='rl_controller',
            executable='odom_pub',
            output='screen',
        ),
       
    ])