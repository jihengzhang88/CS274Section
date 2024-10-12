#!/usr/bin/env python3
import rclpy # type: ignore
from rclpy.node import Node # type: ignore
# import the message type to use
from std_msgs.msg import String # type: ignore


class Publisher(Node):
    def __init__(self):
        super().__init__("publisher_node")
        self.get_logger().info("Publisher has been created!")
        self.pub = self.create_publisher(String, "/chatter", 10)
        self.timer = self.create_timer(0.2, self.publish_msg)
        self.counter = 1

    def publish_msg(self) -> None:
        msg = String()
        msg.data = f"sending constant control… {self.counter}"
        self.pub.publish(msg)
        self.counter += 1

if __name__ == "__main__":
    rclpy.init()        # initialize ROS2 context (must run before any other rclpy call)
    publisher = Publisher()  # instantiate the node
    rclpy.spin(publisher)    # Use ROS2 built-in schedular for executing the node
    rclpy.shutdown()    # cleanly shutdown ROS2 context