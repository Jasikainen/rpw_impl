import math
import numpy as np
import cvxopt as cvx
from visualize_mobile_robot import sim_mobile_robot


# Constants and Settings
Ts = 0.01 # Update simulation
t_max = 30 # total simulation duration in seconds

# Define Field size for plotting (should be in tuple)
field_x = (-3.0, 3.0)
field_y = (-3.0, 3.0)

# general stuff
IS_SHOWING_2DVISUALIZATION = True
OBSTACLE_RADIUS = 0.3
RSI = 0.51              # Rsi for obstacles
RSI2 = math.pow(RSI, 2)  # Rsi²
ROBOT_RADIUS = 0.21

# Maximum velocity of robot
MAX_VELOCITY = 0.18         # m/s

# MAIN SIMULATION COMPUTATION
#---------------------------------------------------------------------
def simulate_control():
    sim_iter = round(t_max/Ts) # Total Step for simulation

    # Initialize robot's state (Single Integrator)
    ROBOT_STATE = np.array([0., 0., 0.])  # px, py, theta

    # Values below change based on the inputs to simulate robot movement
    desired_state = np.array([2.0, 2.0, 0.]) # [px, py, theta]
    obstacle_state = np.array([0.8, 0.4])     # [x, y]
    current_input = np.array([0., 0., 0.]) # [vx, vy, omega]

    if IS_SHOWING_2DVISUALIZATION: # Initialize Plot
        sim_visualizer = sim_mobile_robot( 'omnidirectional' ) # Omnidirectional Icon
        sim_visualizer.set_field(field_x, field_y) # set plot area
        sim_visualizer.show_goal(desired_state)
        sim_visualizer.show_obstacle(obstacle_state, OBSTACLE_RADIUS, RSI)
        
    for it in range(sim_iter):
        # Proportional controller
        u_gtg = 0.1 * (desired_state - ROBOT_STATE)

        # TODO: QP-BASED controller
        #  Use only the x and y values
        # https://courses.csail.mit.edu/6.867/wiki/images/a/a7/Qp-cvxopt.pdf

        # 2 x 2 <-> Q = 2I
        Q = cvx.matrix( np.array([[2, 0], [0,2]]), tc="d") # P
        # 2 x 1 <-> c = -2*u_gtg
        c = -2 * cvx.matrix( np.array([u_gtg[0], u_gtg[1]]), tc="d") # q

        # Fill H and b based on specs - First parameter in array definition is number of constraints / specification
        # n x 2 where row is respect to ux,uy in form: H * u <= b ->  dh/dX * u <= b (NOTE: x != X due to similarity)
        H = cvx.matrix(
            np.array([[ -2 * ROBOT_STATE[0] - obstacle_state[0], -2 * ROBOT_STATE[1] - obstacle_state[1] ]]), tc="d" ) # G

        # Constraints which are used with gamma function to determine how much is u_gtg is altered from the original
        ho1 = np.matmul(np.transpose(ROBOT_STATE[:2] - obstacle_state), ROBOT_STATE[:2] - obstacle_state) - RSI2

        # 1 x 1
        # b = gamma(h(x))
        b = cvx.matrix(np.array([0.2 * ho1]), tc="d") # h

        # Solve the optimization problem
        cvx.solvers.options["show_progress"] = True
        solution = cvx.solvers.qp(Q, c, H, b, verbose=False)
        current_input = np.array([solution['x'][0], solution['x'][1], 0])

        # Limit the velocity to max if necessary
        velocity = np.linalg.norm(current_input[:2])
        if velocity > MAX_VELOCITY:
            # Scale to unit vector by dividing by vectors length and scale to max speed
            current_input[:2] = current_input[:2] / velocity * MAX_VELOCITY

        if IS_SHOWING_2DVISUALIZATION: # Update Plot
            sim_visualizer.update_time_stamp( it*Ts )
            sim_visualizer.update_goal( desired_state )
            sim_visualizer.update_obstacle( obstacle_state, OBSTACLE_RADIUS, RSI )
            sim_visualizer.update_trajectory( np.zeros((1, len(ROBOT_STATE))) )

        # Update the states of obstacle / goal
        #--------------------------------------------------------------------------------
        # Move obstacle based on the controller inputs per time step
        obstacle_state = obstacle_state - Ts * current_input[:2]
        # Move goal closer to turtlebot instead of moving turtlebot towards goal
        desired_state = desired_state - Ts * current_input 

if __name__ == '__main__':
    simulate_control()
