#!/usr/bin/env python3

import rclpy
from asl_tb3_lib.control import BaseController
from asl_tb3_msgs.msg import TurtleBotControl
from std_msgs.msg import Bool

class PerceptionController(BaseController):
    def __init__(self, node_name = "perception_controller") -> None:
        # give it a default node name
        super().__init__(node_name)
        
        self.shutdown_time = None
        self.stop_requested = False
        self.last_start_time = None
        self.declare_parameter("active", True)
        
        self.create_subscription(Bool, "/detector_bool", self.detector_callback, 10)
        
    @property
    def active(self) -> bool:
        return self.get_parameter("active").value
    
    def compute_control(self) -> TurtleBotControl:
        current_time = self.get_clock().now().nanoseconds / 1e9
        
        # Check if robot should be active
        if self.active:
            if self.stop_requested:
                self.shutdown_time = current_time
                self.stop_requested = False  # Reset flag for next detection
                self.set_parameters([rclpy.Parameter(name="active", value=False)])
                return TurtleBotControl(omega=0.0, v=0.0)
            return TurtleBotControl(omega=0.5, v=0.0)  # Spin if no stop requested
        
        else:
            if self.shutdown_time is None:
                self.shutdown_time = current_time
                return TurtleBotControl(omega=0.0, v=0.0)
            else:
                if current_time - self.shutdown_time >= 5.0:
                    self.shutdown_time = None
                    self.set_parameters([rclpy.Parameter(name="active", value=True)])
                    return TurtleBotControl(omega=0.5, v=0.0)
                else:
                    return TurtleBotControl(omega=0.0, v=0.0)
            
    def detector_callback(self, msg) -> None:
        if msg.data and self.active:
            self.stop_requested = True
    
    
    
if __name__ == "__main__":
    rclpy.init()
    node = PerceptionController()
    rclpy.spin(node)
    rclpy.shutdown()
    