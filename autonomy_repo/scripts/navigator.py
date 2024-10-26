#!/usr/bin/env python3

# Import necessary ROS2 packages
import rclpy # type: ignore

# Import specific utilities from the asl_tb3_lib library
import numpy as np
from asl_tb3_lib.navigation import BaseNavigator, TrajectoryPlan
from asl_tb3_msgs.msg import TurtleBotState, TurtleBotControl
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_lib.grids import StochOccupancyGrid2D
from scipy.interpolate import splev, splrep
from P1_astar import AStar
import typing as T

class NavigatorNode(BaseNavigator):
    def __init__(self, node_name = "navigator", kpx=2.0, kpy=2.0, kdx=2.0, kdy=2.0) -> None:
        # give it a default node name
        super().__init__(node_name)

        # Constants for trajectory tracking
        self.V_PREV_THRESH = 0.001
        self.kpx = kpx
        self.kpy = kpy
        self.kdx = kdx
        self.kdy = kdy
        # self.get_logger().info("Navigator node has been started.")
        
        # control gains
        self.kp = 2.0
    
    def reset(self) -> None:
        self.V_prev = 0
        self.om_prev = 0
        self.t_prev = 0
    
    def compute_heading_control(self,
        state: TurtleBotState,
        goal: TurtleBotState,
    ) -> TurtleBotControl:
        control = TurtleBotControl()

        err = wrap_angle(goal.theta - state.theta)
        control.omega = self.kp * err

        return control

    def compute_trajectory_tracking_control(self, state, plan, t) -> T.Tuple[float, float]:
        """
        Inputs:
            x,y,th: Current state
            t: Current time
        Outputs:
            V, om: Control actions
        """
        x_d = splev(t,plan.path_x_spline,der=0)
        xd_d = splev(t,plan.path_x_spline,der=1)
        xdd_d = splev(t,plan.path_x_spline,der=2)
        y_d = splev(t,plan.path_y_spline,der=0)
        yd_d = splev(t,plan.path_y_spline,der=1)
        ydd_d = splev(t,plan.path_y_spline,der=2)
        
        x = state.x
        y = state.y
        th = state.theta
        dt = t - self.t_prev

        V = self.V_prev
    
        # Prevent division by zero
        if abs(V) < self.V_PREV_THRESH:
            self.V_prev = self.V_PREV_THRESH  # Ensure non-zero velocity for omega calculation
        # Compute the current velocities
        Vx = V * np.cos(th)  # x velocity
        Vy = V * np.sin(th)  # y velocity

        # Virtual controls
        u1 = xdd_d + self.kpx * (x_d - x) + self.kdx * (xd_d - Vx)
        u2 = ydd_d + self.kpy * (y_d - y) + self.kdy * (yd_d - Vy)

        # Compute control inputs
        a = u1 * np.cos(th) + u2 * np.sin(th)
        om = (-u1 * np.sin(th) + u2 * np.cos(th)) / V
        
        V = V + a * dt

        # save the commands that were applied and the time
        self.t_prev = t
        self.V_prev = V
        self.om_prev = om

        return TurtleBotControl(v=V, omega = om)

    def compute_trajectory_plan(self,
        state: TurtleBotState,
        goal: TurtleBotState,
        occupancy: StochOccupancyGrid2D,
        resolution: float,
        horizon: float,
    ) -> T.Optional[TrajectoryPlan]:
        """ Compute a trajectory plan using A* and cubic spline fitting

        Args:
            state (TurtleBotState): state
            goal (TurtleBotState): goal
            occupancy (StochOccupancyGrid2D): occupancy
            resolution (float): resolution
            horizon (float): horizon

        Returns:
            T.Optional[TrajectoryPlan]:
        """
        x_0 = state.x
        y_0 = state.y
        world_lo = (x_0 - horizon, y_0 - horizon)
        world_hi = (x_0 + horizon, y_0 + horizon)
        astar = AStar(world_lo, world_hi, (state.x, state.y), (goal.x, goal.y), occupancy, resolution)
        
        if not astar.solve() or len(astar.path) < 4:
            print("No path found! (This is normal, try re-running the block above)")
            return None
        
        self.reset()
        
        astar_path = np.asarray(astar.path)
        plan = self.compute_smooth_plan(astar_path) 
        return plan      
        
    def compute_smooth_plan(self, path, v_desired=0.15, spline_alpha=0.05) -> TrajectoryPlan:
        # Ensure path is a numpy array
        path = np.asarray(path)

        # Compute and set the following variables:
        #   1. ts: 
        #      Compute an array of time stamps for each planned waypoint assuming some constant 
        #      velocity between waypoints. 
        #
        #   2. path_x_spline, path_y_spline:
        #      Fit cubic splines to the x and y coordinates of the path separately
        #      with respect to the computed time stamp array.
        #      Hint: Use scipy.interpolate.splrep
        
        ##### YOUR CODE STARTS HERE #####
        ts_n = np.shape(path)[0]
        ts = np.zeros(ts_n)
        for i in range(ts_n-1):
            ts[i+1] = np.linalg.norm(path[i+1] - path[i]) / v_desired 
            ts[i+1] = ts[i+1] + ts[i]
        # print(ts)
        # print(path[: ,0])
        path_x_spline = splrep(ts, path[: ,0], k=3, s=spline_alpha)
        path_y_spline = splrep(ts, path[: ,1], k=3, s=spline_alpha)
        ###### YOUR CODE END HERE ######
    
        return TrajectoryPlan(
            path=path,
            path_x_spline=path_x_spline,
            path_y_spline=path_y_spline,
            duration=ts[-1],
        )

if __name__ == "__main__":
    rclpy.init()            # initialize ROS client library
    node = NavigatorNode()    # create the node instance
    rclpy.spin(node)        # call ROS2 default scheduler
    rclpy.shutdown()        # clean up after node exits