#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
# import the message type to use
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool


class Publisher(Node):
    def __init__(self):
        super().__init__("publisher_node")
        self.get_logger().info("Publisher has been created!")
        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.sub = self.create_subscription(Bool, "/kill", self.callback, 10)
        self.timer = self.create_timer(0.2, self.publish_msg)

    def publish_msg(self) -> None:      
        msg = Twist()
        msg.linear.x = 0.1
        msg.angular.z = 0.0
        self.pub.publish(msg)
        
    def callback(self, msg):
        if msg.data:
            self.get_logger().fatal("Car has been stoped")
            self.timer.cancel()
            self.pub.publish(Twist())

if __name__ == "__main__":
    rclpy.init()        # initialize ROS2 context (must run before any other rclpy call)
    publisher = Publisher()  # instantiate the node
    rclpy.spin(publisher)    # Use ROS2 built-in schedular for executing the node
    rclpy.shutdown()    # cleanly shutdown ROS2 context