#!/usr/bin/env python3

import numpy as np
import rclpy
import typing as T
from scipy.signal import convolve2d

from std_msgs.msg import Bool

from asl_tb3_msgs.msg import TurtleBotState, TurtleBotControl
from asl_tb3_lib.control import BaseController
from nav_msgs.msg import OccupancyGrid, Path
from asl_tb3_lib.grids import snap_to_grid, StochOccupancyGrid2D

from asl_tb3_lib.control import BaseHeadingController
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_msgs.msg import TurtleBotControl
from asl_tb3_msgs.msg import TurtleBotState
from rclpy.node import Node

class Explorer(Node):
    def __init__(self, node_name: str = "explorer") -> None:
        super().__init__(node_name)

        self.occupancy: T.Optional[StochOccupancyGrid2D] = None
        self.state: T.Optional[TurtleBotState] = None
        self.success: T.Optional[Bool] = None
        self.stop_detected = False  # Track stop sign detection

        self.map_sub = self.create_subscription(OccupancyGrid, "/map", self.map_callback, 10)
        self.state_sub = self.create_subscription(TurtleBotState, "/state", self.state_callback, 10)
        self.nav_suc_sub = self.create_subscription(Bool, "/nav_success", self.send_cmd_nav, 10)
        self.detector_sub = self.create_subscription(Bool, "/detector_bool", self.detector_callback, 10)
        
        self.cmd_nav_pub = self.create_publisher(TurtleBotState, "/cmd_nav", 10)
        self.nav_suc_pub = self.create_publisher(Bool, "/nav_success", 10)
        
        self.declare_parameter("window_size", 13)
        self.declare_parameter("perturbation", 1e-1)

    @property
    def window_size(self) -> float:
        return self.get_parameter("window_size").value
    
    @property
    def perturbation(self) -> float:
        return self.get_parameter("perturbation").value

    def detector_callback(self, msg: Bool) -> None:
        """Handle stop sign detection."""
        if msg.data and not self.stop_detected:
            self.stop_detected = True
            self.get_logger().info("Stop sign detected. Pausing for 5 seconds.")
            
            # Create a one-shot timer that will trigger after 5 seconds
            self.cmd_nav_pub.publish(self.state)
            self.resume_timer = self.create_timer(5.0, self.resume_exploration)
            
            
    def resume_exploration(self):
        self.get_logger().info("Resuming exploration.")
        self.stop_detected = False
        # Destroy the timer to free resources
        self.resume_timer.destroy()
        bool = Bool()
        bool.data = True
        self.nav_suc_pub.publish(bool)

    def state_callback(self, msg: TurtleBotState) -> None:
        self.state = msg

    def explore(self) -> TurtleBotState:        
        convArray = np.ones([self.window_size, self.window_size])
        probs = self.occupancy.probs
        opGrid = np.zeros(probs.shape)
        unGrid = np.zeros_like(opGrid)
        occGrid = np.zeros_like(opGrid)
        for i in range(opGrid.shape[0]):
            for j in range(opGrid.shape[1]):
                if probs[i,j] >= 0.5:
                    occGrid[i,j] = 1
                elif probs[i,j] >= 0:
                    opGrid[i,j] = 1
                elif probs[i,j] == -1:
                    unGrid[i,j] = 1

        occAmt = convolve2d(occGrid, convArray, 'same')
        opAmt = convolve2d(opGrid, convArray, 'same')
        unAmt = convolve2d(unGrid, convArray, 'same')

        W = self.window_size**2

        occThresh = 0
        unThresh = 0.2*W
        opThresh = 0.3*W

        frontier_states = []

        for i in range(occGrid.shape[0]):
            for j in range(occGrid.shape[1]):
                if unAmt[i, j] >= unThresh and opAmt[i, j] >= opThresh and occAmt[i, j] == occThresh:
                    state = self.occupancy.grid2state(np.array([[j,i]]))
                    frontier_states.append(state)

        try:
            frontier_states = np.concatenate(frontier_states)
        except:
            self.get_logger().info(f"All frontiers explored")
            return self.state

        current_state = np.array([self.state.x, self.state.y])

        dist = np.min(np.linalg.norm(current_state - frontier_states, axis = 1))
        idx = np.argmin(np.linalg.norm(current_state - frontier_states, axis = 1))

        frontier = frontier_states[idx]

        front_state = TurtleBotState(x=frontier[0], y=frontier[1], theta=self.state.theta)

        return front_state
    
    def publish_cmd_nav(self, msg: TurtleBotState) -> None:
        if not self.stop_detected:
            self.cmd_nav_pub.publish(msg)

    def map_callback(self, msg: OccupancyGrid) -> None:
        self.occupancy = StochOccupancyGrid2D(
            resolution=msg.info.resolution,
            size_xy=np.array([msg.info.width, msg.info.height]),
            origin_xy=np.array([msg.info.origin.position.x, msg.info.origin.position.y]),
            window_size=self.window_size,
            probs=msg.data,
        )

        if self.success is None and not(self.state is None):
            self.frontier = TurtleBotState(x = self.state.x, y = self.state.y, theta = self.state.theta)
            nav_msg = Bool()
            nav_msg.data = True    
            self.send_cmd_nav(nav_msg)
    
    def send_cmd_nav(self, nav_msg: Bool) -> None:
        if not self.stop_detected:
            cmd_msg = self.explore()
            self.frontier = cmd_msg
            self.publish_cmd_nav(self.frontier)
            self.success = nav_msg

if __name__ == "__main__":
    rclpy.init()
    explorer = Explorer()
    rclpy.spin(explorer)
    rclpy.shutdown()