import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Define the path to the map file
    map_dir = LaunchConfiguration(
        'maps',
        default=os.path.join(
            get_package_share_directory('rl_slam'),
            'maps',
            'map_flood4x15.yaml'))  # Replace with your map file

    lifecycle_nodes = ['map_server']

    return LaunchDescription([
        # Launch argument to specify the map file
        DeclareLaunchArgument(
            'maps',
            default_value=map_dir,
            description='Full path to map file to load'),

        # Map server node
        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            output='screen',
            parameters=[{'use_sim_time': True},
                        {'yaml_filename': map_dir}]),

        # # Static transform publisher to ensure the transform is available
        # Node(
        #     package='tf2_ros',
        #     executable='static_transform_publisher',
        #     arguments=['0.0', '0', '0.0', '0', '0', '0', 'map', 'odom'],
        #     output='screen'
        # ),

        # Lifecycle manager to manage the map_server node
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_mapper',
            output='screen',
            parameters=[{'use_sim_time': True},
                        {'autostart': True},
                        {'node_names': lifecycle_nodes}]
        ),
    ])
