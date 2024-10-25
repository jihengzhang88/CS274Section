#!/usr/bin/env python3

# Import necessary ROS2 packages
import rclpy
from rclpy.node import Node

# Import specific utilities from the asl_tb3_lib library
from asl_tb3_lib.navigation import BaseNavigator, TrajectoryPlan
from asl_tb3_msgs.msg import TurtleBotState, TurtleBotControl
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_lib.tf_utils import quaternion_to_yaw
from scipy.interpolate import splev, splprep
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_line_segments(segments, **kwargs):
    plt.plot([x for tup in [(p1[0], p2[0], None) for (p1, p2) in segments] for x in tup],
             [y for tup in [(p1[1], p2[1], None) for (p1, p2) in segments] for y in tup], **kwargs)

def compute_spline_params(path):
    """
    Takes a path (list of (x, y) tuples) and returns spline parameters.
    This is an approximation function for generating a smooth trajectory.
    """
    # Separate x and y coordinates
    x_points = [point[0] for point in path]
    y_points = [point[1] for point in path]

    # Generate spline parameters for x and y coordinates
    # s parameter can be tuned for smoothness; try s=0 for interpolation
    tck, _ = splprep([x_points, y_points], s=0)
    return tck

class AStar(object):
    """Represents a motion planning problem to be solved using A*"""

    def __init__(self, statespace_lo, statespace_hi, x_init, x_goal, occupancy, resolution=1):
        self.statespace_lo = statespace_lo         # state space lower bound (e.g., [-5, -5])
        self.statespace_hi = statespace_hi         # state space upper bound (e.g., [5, 5])
        self.occupancy = occupancy                 # occupancy grid (a DetOccupancyGrid2D object)
        self.resolution = resolution               # resolution of the discretization of state space (cell/m)
        self.x_offset = x_init                     
        self.x_init = self.snap_to_grid(x_init)    # initial state
        self.x_goal = self.snap_to_grid(x_goal)    # goal state

        self.closed_set = set()    # the set containing the states that have been visited
        self.open_set = set()      # the set containing the states that are condidate for future expension

        self.est_cost_through = {}  # dictionary of the estimated cost from start to goal passing through state (often called f score)
        self.cost_to_arrive = {}    # dictionary of the cost-to-arrive at state from start (often called g score)
        self.came_from = {}         # dictionary keeping track of each state's parent to reconstruct the path

        self.open_set.add(self.x_init)
        self.cost_to_arrive[self.x_init] = 0
        self.est_cost_through[self.x_init] = self.distance(self.x_init,self.x_goal)

        self.path = None        # the final path as a list of states

    def is_free(self, x):
        """
        Checks if a give state x is free, meaning it is inside the bounds of the map and
        is not inside any obstacle.
        Inputs:
            x: state tuple
        Output:
            Boolean True/False
        Hint: self.occupancy is a DetOccupancyGrid2D object, take a look at its methods for what might be
              useful here
        """
        ########## Code starts here ##########
        if x==self.x_init or x==self.x_goal:
            return True
        for dim in range(len(x)):
            if x[dim] < self.statespace_lo[dim]:
                return False
            if x[dim] > self.statespace_hi[dim]:
                return False
        if not self.occupancy.is_free(x):
            return False
        return True
        ########## Code ends here ##########

    def distance(self, x1, x2):
        """
        Computes the Euclidean distance between two states.
        Inputs:
            x1: First state tuple
            x2: Second state tuple
        Output:
            Float Euclidean distance

        HINT: This should take one line. Tuples can be converted to numpy arrays using np.array().
        """
        ########## Code starts here ##########
        return np.linalg.norm(np.array(x1)-np.array(x2))
        ########## Code ends here ##########

    def snap_to_grid(self, x):
        """ Returns the closest point on a discrete state grid
        Input:
            x: tuple state
        Output:
            A tuple that represents the closest point to x on the discrete state grid
        """
        return (
            self.resolution * round((x[0] - self.x_offset[0]) / self.resolution) + self.x_offset[0],
            self.resolution * round((x[1] - self.x_offset[1]) / self.resolution) + self.x_offset[1],
        )

    def get_neighbors(self, x):
        """
        Gets the FREE neighbor states of a given state x. Assumes a motion model
        where we can move up, down, left, right, or along the diagonals by an
        amount equal to self.resolution.
        Input:
            x: tuple state
        Ouput:
            List of neighbors that are free, as a list of TUPLES

        HINTS: Use self.is_free to check whether a given state is indeed free.
               Use self.snap_to_grid (see above) to ensure that the neighbors
               you compute are actually on the discrete grid, i.e., if you were
               to compute neighbors by adding/subtracting self.resolution from x,
               numerical errors could creep in over the course of many additions
               and cause grid point equality checks to fail. To remedy this, you
               should make sure that every neighbor is snapped to the grid as it
               is computed.
        """
        neighbors = []
        ########## Code starts here ##########
        for dx1 in [-self.resolution, 0, self.resolution]:
            for dx2 in [-self.resolution, 0, self.resolution]:
                if dx1==0 and dx2==0:
                    # don't include itself
                    continue
                new_x = (x[0]+dx1,x[1]+dx2)
                if self.is_free(new_x):
                    neighbors.append(self.snap_to_grid(new_x))
        ########## Code ends here ##########
        return neighbors

    def find_best_est_cost_through(self):
        """
        Gets the state in open_set that has the lowest est_cost_through
        Output: A tuple, the state found in open_set that has the lowest est_cost_through
        """
        return min(self.open_set, key=lambda x: self.est_cost_through[x])

    def reconstruct_path(self):
        """
        Use the came_from map to reconstruct a path from the initial location to
        the goal location
        Output:
            A list of tuples, which is a list of the states that go from start to goal
        """
        path = [self.x_goal]
        current = path[-1]
        while current != self.x_init:
            path.append(self.came_from[current])
            current = path[-1]
        return list(reversed(path))

    def plot_path(self, fig_num=0, show_init_label=True):
        """Plots the path found in self.path and the obstacles"""
        if not self.path:
            return

        self.occupancy.plot(fig_num)

        solution_path = np.asarray(self.path)
        plt.plot(solution_path[:,0],solution_path[:,1], color="green", linewidth=2, label="A* solution path", zorder=10)
        plt.scatter([self.x_init[0], self.x_goal[0]], [self.x_init[1], self.x_goal[1]], color="green", s=30, zorder=10)
        if show_init_label:
            plt.annotate(r"$x_{init}$", np.array(self.x_init) + np.array([.2, .2]), fontsize=16)
        plt.annotate(r"$x_{goal}$", np.array(self.x_goal) + np.array([.2, .2]), fontsize=16)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), fancybox=True, ncol=3)

        plt.axis([0, self.occupancy.width, 0, self.occupancy.height])

    def plot_tree(self, point_size=15):
        plot_line_segments([(x, self.came_from[x]) for x in self.open_set if x != self.x_init], linewidth=1, color="blue", alpha=0.2)
        plot_line_segments([(x, self.came_from[x]) for x in self.closed_set if x != self.x_init], linewidth=1, color="blue", alpha=0.2)
        px = [x[0] for x in self.open_set | self.closed_set if x != self.x_init and x != self.x_goal]
        py = [x[1] for x in self.open_set | self.closed_set if x != self.x_init and x != self.x_goal]
        plt.scatter(px, py, color="blue", s=point_size, zorder=10, alpha=0.2)

    def solve(self):
        """
        Solves the planning problem using the A* search algorithm. It places
        the solution as a list of tuples (each representing a state) that go
        from self.x_init to self.x_goal inside the variable self.path
        Input:
            None
        Output:
            Boolean, True if a solution from x_init to x_goal was found

        HINTS:  We're representing the open and closed sets using python's built-in
                set() class. This allows easily adding and removing items using
                .add(item) and .remove(item) respectively, as well as checking for
                set membership efficiently using the syntax "if item in set".
        """
        ########## Code starts here ##########
        while len(self.open_set)>0:
            current = self.find_best_est_cost_through()
            if current == self.x_goal:
                self.path = self.reconstruct_path()
                return True
            self.open_set.remove(current)
            self.closed_set.add(current)
            for n in self.get_neighbors(current):
                if n in self.closed_set:
                    continue
                tentative_cost_to_arrive = self.cost_to_arrive[current] + self.distance(current,n)
                if n not in self.open_set:
                    self.open_set.add(n)
                elif tentative_cost_to_arrive >= self.cost_to_arrive[n]:
                    continue
                self.came_from[n] = current
                self.cost_to_arrive[n] = tentative_cost_to_arrive
                self.est_cost_through[n] = self.cost_to_arrive[n] + self.distance(n,self.x_goal)

        return False
        ########## Code ends here ##########


class NavigatorNode(BaseNavigator):
    def __init__(self) -> None:
        # give it a default node name
        super().__init__("navigator_node")

        # Constants for trajectory tracking
        self.V_PREV_THRESH = 0.5

        self.get_logger().info("Navigator node has been started.")
        
        # control gains
        self.declare_parameter("kp", 2.0)

    @property
    def kp(self) -> float:
        return self.get_parameter("kp").value
    
    def compute_heading_control(self,
        state: TurtleBotState,
        goal: TurtleBotState,
    ) -> TurtleBotControl:
        control = TurtleBotControl()

        err = wrap_angle(goal.theta - state.theta)
        control.omega = self.kp * err

        return control
    
    def compute_trajectory_tracking_control(self, state, trajectory_plan):
        t = state.t
        x_d, xd_d, xdd_d, y_d, yd_d, ydd_d = splev(t, trajectory_plan.spline_params)

        # Desired position, velocity, and acceleration
        position_error = np.array([x_d - state.x, y_d - state.y])
        velocity_error = np.array([xd_d - state.v_x, yd_d - state.v_y])

        # Control gains (assuming they're proportional gains; adjust as needed)
        kp = 1.0
        kd = 0.1

        # Proportional-Derivative control for trajectory tracking
        u_x = kp * position_error[0] + kd * velocity_error[0]
        u_y = kp * position_error[1] + kd * velocity_error[1]

        # Construct control based on state
        control = TurtleBotControl(u_x, u_y)
        return control

    def compute_trajectory_plan(self, start, goal):
        # Initialize AStar with map parameters
        astar = AStar(self.map, start, goal)
        solution_path = astar.solve()

        if not solution_path:
            self.get_logger().error("No path found.")
            return None

        # Smooth the path using spline fitting
        x_smooth, y_smooth = compute_spline_params(solution_path)
        spline_params = (x_smooth, y_smooth)

        # Return a TrajectoryPlan with the smoothed path
        return TrajectoryPlan(spline_params)

if __name__ == "__main__":
    rclpy.init()            # initialize ROS client library
    node = NavigatorNode()    # create the node instance
    rclpy.spin(node)        # call ROS2 default scheduler
    rclpy.shutdown()        # clean up after node exits