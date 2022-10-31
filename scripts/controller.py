import rospy
import cvxopt
import numpy as np
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rpw_impl.msg import ObstacleData, ObstacleArray
from tf.transformations import euler_from_quaternion

ERROR_MARGIN = 0.1
GOAL = [2.0,1.0,0.0]


def get_yaw(orientation):
    (_,_,yaw) = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
    return yaw


class QpController:
    def __init__(self):
        rospy.init_node('turtlebot_controller')
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.update_goal)
        self.obj_sub = rospy.Subscriber('/obstacles', ObstacleArray, self.callback)
        self.lx = 0.0
        self.az = 0.0
        self.goal = GOAL
        self.relative_goal = []
        #self.obstacle_centers = []
        #self.obstacle_radii = []
        self.obstacle1_center = []
        self.obstacle2_center = []
        self.obstacle1_r = 0.0
        self.obstacle2_r = 0.0
        self.safety_margin = 0.1
        rospy.spin()


    def callback(self,data):
        self.goal = GOAL
        obstacles = sorted(data.obstacles, key=lambda d: d.distance)
        if len(obstacles) == 0:
            return
        """ TODO: Handle N-amount of obstacles
        for obstacle in obstacles:
            self.obstacle_centers = [obstacle.center[0],obstacle.center[1]]
            self.obstacle_radii = obstacle.radius
        """
        # For now, we calculate based on the two closest obstacles    
        self.obstacle1_center = [obstacles[0].center[0],obstacles[0].center[1]]
        self.obstacle1_r = obstacles[0].radius + self.safety_margin
        if len(obstacles) >= 2:
            self.obstacle2_center = [obstacles[1].center[0],obstacles[1].center[1]]
            self.obstacle2_r = obstacles[1].radius + self.safety_margin
        else:
            # Only one obstacle found
            self.obstacle2_center = [1000,1000]
            self.obstacle2_r = self.obstacle1_r

        self.solve_twist()


    def solve_twist(self):
        error_dist = np.linalg.norm(self.relative_goal)
        if error_dist < ERROR_MARGIN:
            self.pub_twist(True)
            return
        v_0 = 0.22
        beta = 3
        k = v_0*(1-np.exp(-beta*error_dist))/error_dist
        u_gtg = k * np.array(self.relative_goal)

        # QP-based controller
        gamma = 0.5
        l = 0.1 # "Caster wheel" distance from turtle center

        Q_mat = 2 * cvxopt.matrix(np.eye(2), tc='d')
        c_mat = -2 * cvxopt.matrix(u_gtg[:2], tc='d')

        h_o1 = np.linalg.norm(np.array(self.obstacle1_center))**2 - self.obstacle1_r**2
        h_o2 = np.linalg.norm(np.array(self.obstacle2_center))**2 - self.obstacle2_r**2
        dh_o1 = -2*np.transpose(np.negative(self.obstacle1_center))
        dh_o2 = -2*np.transpose(np.negative(self.obstacle2_center))
        H = np.array([dh_o1, dh_o2])
        b = gamma*np.array([[h_o1],[h_o2]])

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
        
        self.lx = v[0]
        self.az = w[0]
        print(f"single integrator u: {u[0]:>5.2f}, {u[1]:>5.2f}")
        print(f"theta: {np.rad2deg(theta):>5.2f} deg")
        self.pub_twist()


    def pub_twist(self, stop=False):
        cmd = Twist()
        if stop:
            self.cmd_vel_pub.publish(cmd)
            return
        cmd.linear.x = self.lx
        cmd.angular.z = self.az
        if cmd.linear.x > 0.22:
            cmd.linear.x = 0.22
        elif cmd.linear.x < -0.22:
            cmd.linear.x = -0.22
        if cmd.angular.z > 1:
            cmd.angular.z = 1
        elif cmd.angular.z < -1:
            cmd.angular.z = -1

        print(f"\nX: {cmd.linear.x:.2f}, Z: {cmd.angular.z:.2f}")
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

