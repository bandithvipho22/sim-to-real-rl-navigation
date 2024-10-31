from setuptools import find_packages, setup
import os 
from glob import glob

package_name = 'rl_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name,'launch'), glob('rl_launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='vipho',
    maintainer_email='you@example.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'astar = rl_controller.astar:main',
            'imu_pub = rl_controller.imu_pub:main',
            'odom_pub = rl_controller.odom_pub:main',
            'pure_pursuit = rl_controller.pure_pursuit:main',
        ],
    },
)
