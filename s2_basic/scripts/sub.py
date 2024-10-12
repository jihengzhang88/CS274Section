#!/usr/bin/env python3
import rclpy # type: ignore
from rclpy.node import Node # type: ignore
# import the message type to use
from std_msgs.msg import String # type: ignore


class Subscriber(Node):
    def __init__(self):
        super().__init__("subscriber_node")
        self.get_logger().info("Subscriber has been created!")
        self.sub = self.create_subscription(String, "/chatter", self.callback, 10)
        self.counter = 1

    def callback(self, msg):
        data = msg.data
        self.get_logger().info(f"I heard: {data}")

if __name__ == "__main__":
    rclpy.init()        # initialize ROS2 context (must run before any other rclpy call)
    subscriber = Subscriber()  # instantiate the heartbeat node
    rclpy.spin(subscriber)    # Use ROS2 built-in schedular for executing the node
    rclpy.shutdown()    # cleanly shutdown ROS2 context