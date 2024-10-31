from setuptools import find_packages, setup
import os 
from glob import glob

package_name = 'rl_gazebo'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name,'launch'), glob('rl_launch/*.launch.py')),
        
        (os.path.join('share', package_name,'worlds/'), glob('./worlds/*')),
        
        # Path to the pioneer sdf file
        (os.path.join('share', package_name,'models/pioneer3at/'), glob('./models/pioneer3at/model.sdf')),

        # Path to the pioneer config file
        (os.path.join('share', package_name,'models/pioneer3at/'), glob('./models/pioneer3at/model.config')),
        
        # Path to the target sdf file
        (os.path.join('share', package_name,'models/Target/'), glob('./models/Target/model.sdf')),

        # Path to the target config file
        (os.path.join('share', package_name,'models/Target/'), glob('./models/Target/model.config')),
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
            'spawn_diffbot = rl_gazebo.spawn_diffbot:main',
        ],
    },
)
