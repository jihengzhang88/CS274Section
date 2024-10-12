#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
# import the message type to use
from geometry_msgs.msg import Twist


class Publisher(Node):
    def __init__(self):
        super().__init__("publisher_node")
        self.get_logger().info("Publisher has been created!")
        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.timer = self.create_timer(0.2, self.publish_msg)
        self.counter = 1
        self.msg = Twist()
        self.msg.linear.x = 0.0
        self.msg.angular.z = 300.0

    def publish_msg(self) -> None:      
        if self.counter % 5 == 0.0:
            self.msg.angular.z -= 50.0
        self.pub.publish(self.msg)
        self.counter += 1

if __name__ == "__main__":
    rclpy.init()        # initialize ROS2 context (must run before any other rclpy call)
    publisher = Publisher()  # instantiate the node
    rclpy.spin(publisher)    # Use ROS2 built-in schedular for executing the node
    rclpy.shutdown()    # cleanly shutdown ROS2 context