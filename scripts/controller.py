import rospy
import cvxopt
import numpy as np
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rpw_impl.msg import ObstacleData, ObstacleArray
from tf.transformations import euler_from_quaternion
from enum import Enum

ERROR_MARGIN = 0.1
GOAL = [3.0,1.0,0.0]
MAX_LINEAR_VEL = 0.22 # 0.22 m/s for Turtlebot3 Burger
MULTIPLE_GOALS = {
     0: [0.0, -1.6, 0.0], 
     1: [1.5, 0.5, 0.0],
     2: [-2.0, 0.5, 0.0],
     3: [-2.0, -0.5, 0.0]}

def get_yaw(orientation):
    (_,_,yaw) = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
    return yaw

class GammaFunctionType(Enum):
    LINEAR = 1
    QUADRATIC = 2
    CUBIC = 3

"""
Inputs:
h:              Value calculated based on obstacle shape it's distance between robot
multiplier :    Coefficient used in calculation
function_type : Type of gamma function used

Output:
Value of gamma function computed or 0.0 if function_type not supported (see: GammaFunctionType Enum)
"""
def gamma(h, multiplier, function_type):
    if function_type == GammaFunctionType.LINEAR:
        return multiplier * h
    elif function_type == GammaFunctionType.QUADRATIC:
        return multiplier * h**2
    elif function_type == GammaFunctionType.CUBIC:
        return multiplier * h**3
    return 0.0
            

class QpController:
    def __init__(self):
        rospy.init_node('turtlebot_controller')
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.update_goal)
        self.obj_sub = rospy.Subscriber('/obstacles', ObstacleArray, self.callback)
        self.linear_x = 0.0
        self.angular_z = 0.0
        self.goal_index = 0
        self.goal = MULTIPLE_GOALS[self.goal_index]
        self.relative_goal = []
        self.obstacle_centers = []
        self.obstacle_radii = []
        self.safety_margin = 0.1
        self.gamma_function_type = GammaFunctionType.CUBIC
        rospy.spin()


    def callback(self,data):
        self.goal = MULTIPLE_GOALS[self.goal_index]
        obstacles = sorted(data.obstacles, key=lambda d: d.distance)
        if len(obstacles) == 0:
            return

        # Handle multiple obstacles published to the topic ObstacleArray
        if self.obstacle_centers:
            self.obstacle_centers.clear()
        if self.obstacle_radii:
            self.obstacle_radii.clear()

        for obstacle in obstacles:
            self.obstacle_centers.append([obstacle.center[0], obstacle.center[1]]) 
            self.obstacle_radii.append(obstacle.radius + self.safety_margin)

        self.solve_twist()


    def solve_twist(self):
        error_dist = np.linalg.norm(self.relative_goal)
        # Goal reached, time to change the goal
        if error_dist < ERROR_MARGIN:
            self.pub_twist(True)
            if self.goal_index < len(MULTIPLE_GOALS) - 1: # Stop updating goal at last available goal
                self.goal_index += 1
                self.goal = MULTIPLE_GOALS[self.goal_index]
            return

        print(f"Distance {error_dist:.2f} to relative goal (x,y) = ({self.relative_goal[0]}, {self.relative_goal[1]})")

        v_0 = MAX_LINEAR_VEL
        beta = 3
        k = v_0*(1-np.exp(-beta*error_dist))/error_dist
        u_gtg = k * np.array(self.relative_goal)

        # QP-based controller
        # -------------------
        # "Caster wheel" distance from turtle center
        l = 0.06 

        Q_mat = 2 * cvxopt.matrix(np.eye(2), tc='d')
        c_mat = -2 * cvxopt.matrix(u_gtg[:2], tc='d')

        # Allocate space based on the number of found obstacles
        n = len(self.obstacle_centers)
        H = np.zeros( (n,2) ) # n x 2
        b = np.zeros( (n,1) ) # n x 1

        for index, (center, radius) in enumerate(zip(self.obstacle_centers, self.obstacle_radii)):
            # Calculate the gamma(h(x)) and add to b matrix
            h_obstacle = np.linalg.norm(np.array(center))**2 - radius**2
            b[index] = gamma(h_obstacle, 10, self.gamma_function_type)
            # Calculate the rows of H matrix
            dh_obstacle = -2 * np.transpose(np.negative(center))
            H[index] = dh_obstacle
        
        print(f"H: {np.shape(H)}\nb: {np.shape(b)}:")
        # Format matrices in suitable type for cvxopt solver
        H_mat = cvxopt.matrix(H, tc='d')
        b_mat = cvxopt.matrix(b, tc='d')

        cvxopt.solvers.options['show_progress'] = False
        sol = cvxopt.solvers.qp(Q_mat, c_mat, H_mat, b_mat, verbose=False)

        u = np.array([sol['x'][0], sol['x'][1], 0])
        theta = np.arctan2(u[1],u[0])
        A = np.array([[1,0], [0,1/l]]) @ np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        [v, w] = A @ np.array([[u[0]],[u[1]]]) # Using the data from the omnidirectional case
        
        # Quick fix for turtle stuck going towards +-180 deg
        if abs(theta) > np.pi/2:
            w[0] = -w[0]
        
        self.linear_x = v[0]
        self.angular_z = w[0]
        print(f"single integrator ux: {u[0]:>5.2f} uy: {u[1]:>5.2f}")
        print(f"theta: {np.rad2deg(theta):>5.2f} deg")
        self.pub_twist()


    def pub_twist(self, stop=False):
        cmd = Twist()
        if stop:
            self.cmd_vel_pub.publish(cmd)
            return
        cmd.linear.x = self.linear_x
        cmd.angular.z = self.angular_z
        if cmd.linear.x > MAX_LINEAR_VEL:
            cmd.linear.x = MAX_LINEAR_VEL
        elif cmd.linear.x < -MAX_LINEAR_VEL:
            cmd.linear.x = -MAX_LINEAR_VEL
        if cmd.angular.z > 1:
            cmd.angular.z = 1
        elif cmd.angular.z < -1:
            cmd.angular.z = -1

        print(f"\n{'Velocity':.^20} \nLinear: {cmd.linear.x:.2f} Angular: {cmd.angular.z:.2f}")
        self.cmd_vel_pub.publish(cmd)

    
    def update_goal(self,data):
        posx = data.pose.pose.position.x
        posy = data.pose.pose.position.y
        yaw = get_yaw(data.pose.pose.orientation)
        current_pose = np.array([posx,posy])
        goal_delta = np.subtract(self.goal[:2],current_pose)
        inv_rot_matrix = np.array([[np.cos(yaw),np.sin(yaw)],[-np.sin(yaw),np.cos(yaw)]])
        self.relative_goal = inv_rot_matrix @ np.array(goal_delta)

if __name__ == '__main__':
    controller = QpController()
    cmd = Twist()
    cmd.linear.x = 0
    cmd.angular.z = 0
    controller.cmd_vel_pub.publish(cmd)

