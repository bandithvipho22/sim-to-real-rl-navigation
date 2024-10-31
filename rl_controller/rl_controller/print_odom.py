import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry

class OdomPosition(Node):
    def __init__(self):
        super().__init__('odom_position_node')
        self.subscription = self.create_subscription(
            Odometry,
            '/demo/odom',
            self.odom_callback,
            10
        )
        self.subscription  # prevent unused variable warning

    def odom_callback(self, msg):
        position = msg.pose.pose.position
        self.get_logger().info(f'Position: x={position.x}, y={position.y}, z={position.z}')


def main(args=None):
    rclpy.init(args=args)
    node = OdomPosition()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

