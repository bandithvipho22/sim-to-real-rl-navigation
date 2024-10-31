"""
Demo for spawn_entity.
Launches Gazebo and spawns a model
"""
# A bunch of software packages that are needed to launch ROS2
import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import ThisLaunchFileDir,LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='True')
    world_file_name = 'floor4x15.world'
    pkg_dir = get_package_share_directory('rl_gazebo')

    os.environ["GAZEBO_MODEL_PATH"] = os.path.join(pkg_dir, 'models')

    world = os.path.join(pkg_dir, 'worlds', world_file_name)
    launch_file_dir = os.path.join(pkg_dir, 'launch')

    gazebo = ExecuteProcess(
            cmd=['gzserver', '--verbose', world, '-s', 'libgazebo_ros_init.so',
            '-s', 'libgazebo_ros_factory.so'],
            output='screen')

    # GAZEBO_MODEL_PATH has to be correctly set for Gazebo to be able to find the model
    #spawn_entity = Node(package='gazebo_ros', node_executable='spawn_entity.py',
    #                    arguments=['-entity', 'demo', 'x', 'y', 'z'],
    #                    output='screen')
    spawn_entity = Node(package='rl_gazebo', executable='spawn_diffbot',
                        # arguments=['HospitalBot', 'demo', '1', '16.0', '0.0'],
                        arguments=['DiffBot', 'demo', '-7.76', '2.17', '0.0'],
                        # arguments=['HospitalBot', 'demo', '0.0', '0.0', '0.0'],
                        output='screen')
    
    # Add static transform publisher
    static_transform_publisher = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'chassis', 'laser_link']
    )
    
    static_transform_publisher2 = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'odom', 'map']
    )
    
    static_transform_publisher3 = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'laser_link']
    )

    return LaunchDescription([
        gazebo,
        spawn_entity,
        static_transform_publisher,
        static_transform_publisher2,
        # static_transform_publisher3
    ])
