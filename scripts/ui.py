import rospy
import time
import argparse
from sensor_msgs.msg import LaserScan
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrow, Polygon
from matplotlib.backend_bases import MouseButton as mb
from rpw_impl.msg import ObstacleData, ObstacleArray, SIControlOutput
from geometry_msgs.msg import Point
from _tkinter import TclError


parser = argparse.ArgumentParser()
parser.add_argument("--namespace",
            help="prepend all topics with this namespace",
            default=rospy.get_namespace().rstrip("/"))
args = parser.parse_known_args()

NAMESPACE = args[0].namespace.rstrip("/")
DRAW_ESTIMATED_OBSTACLES = True
UPDATE_RATE = 0.001
SAFETY_MARGIN = 0.3
# Globals
scan_points = ([],[])
obstacles = []
relative_goal = (0.0,0.0) # From controller
goal_pub = rospy.Publisher(NAMESPACE+"/new_goal", Point, queue_size=10)
control_output = (0.0,0.0) # From controller

class TurtlebotUI:
    def __init__(self,AxisLimits=[-3,3,-2,2]):
        plt.ion()
        self.al = AxisLimits
        self.fig = plt.figure(num="Turtlebot UI", facecolor="Black",figsize=[8.0,8.0])
        self.fig.set_tight_layout(True)
        self.ax = self.fig.gca()
        self.ax.set_aspect('equal')
        self.ax.set_facecolor("Black")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_xlim(self.al[0],self.al[1])
        self.ax.set_ylim(self.al[2],self.al[3])
        for spine in self.ax.spines.values():
            spine.set_edgecolor('gray')

        self.draw_turtlebot()
        self.dynamic_objects = []

    def draw_turtlebot(self):
        turtleBodyPts = [(.04,.08),(-.04,.08),(-.08,.04),(-.08,-.04),(-.04,-.08),(.04,-.08)]
        turtleLidarSensorPts = [(-.03,.03),(-.03,-.03),(.05,-.01),(.05,.01)]
        turtleLeftWheelPts = [(.04,.08),(.04,.1),(0,.1),(0,.08)]
        turtleRightWheelPts = [(.04,-.08),(.04,-.1),(0,-.1),(0,-.08)]
        turtleBody = Polygon(turtleBodyPts,closed=True, color="gray", fill=True, zorder=100)
        turtleLidarSensor = Polygon(turtleLidarSensorPts, color=(.1,.1,.1), fill=True, zorder=101)
        turtleLeftWheel = Polygon(turtleLeftWheelPts,closed=True,color="gray",fill=False,zorder=102)
        turtleRightWheel = Polygon(turtleRightWheelPts,closed=True,color="gray",fill=False,zorder=103)
        turtlePointAhead = Circle((0.06,0),radius=0.02,color="white",fill=True,zorder=104)
        self.ax.add_patch(turtleBody)
        self.ax.add_patch(turtleLidarSensor)
        self.ax.add_patch(turtleLeftWheel)
        self.ax.add_patch(turtleRightWheel)
        self.ax.add_patch(turtlePointAhead)

    def refresh(self):
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        [p.remove() for p in self.dynamic_objects]
        self.dynamic_objects = []

    def update_plot(self, plot_obstacles=False):
        # Single integrator distance from robot body
        lx = 0.06

        # Plot lidar data points
        scatter = self.ax.scatter(scan_points[0],scan_points[1],s=10.0,c="Red")
        self.dynamic_objects.append(scatter)

        # Plot estimated obstacles
        if plot_obstacles:
            for obstacle in obstacles:
                circle = Circle(xy=obstacle.center, radius=obstacle.radius, color="Green",fill=True,alpha=0.3, zorder=0)
                circle_with_margin = Circle(xy=obstacle.center, radius=obstacle.radius + SAFETY_MARGIN, color="Green",fill=False, alpha=0.1, zorder=0)
                if obstacle.name == "-1":
                    # Closest point from the "noise" cluster
                    circle.set_color("yellow")
                    circle.set_alpha(.4)
                self.dynamic_objects.append(self.ax.add_patch(circle))
                self.dynamic_objects.append(self.ax.add_patch(circle_with_margin))
                
        # Plot goal
        distance_to_goal = np.linalg.norm(relative_goal)
        if distance_to_goal > 0.1:
            goal = FancyArrow(lx,0,relative_goal[0]-lx,relative_goal[1],width=.03, linewidth=0,
                                length_includes_head=True, color="Yellow", alpha=.25, zorder=1)
            arrow_goal = self.ax.add_patch(goal)
            self.dynamic_objects.append(arrow_goal)
            self.ax.set_title(f"Distance to goal: {distance_to_goal:.2f} meters", color="gray")
        else:
            self.ax.set_title(" ",color="black")

        # Plot control output
        if control_output[0] != 0.0:
            # Scale QP controller output vector length to same as distance to goal
            control_output_scaled = distance_to_goal / np.linalg.norm(control_output) * np.array([control_output[0], control_output[1]])
            control = FancyArrow(lx,0,control_output_scaled[0]-lx,control_output_scaled[1],width=.02, linewidth=0,
                                length_includes_head=True, color="Green", alpha=.25, zorder=1)
            arrow_control_output = self.ax.add_patch(control)
            self.dynamic_objects.append(arrow_control_output)

        self.refresh()


def callback_scan(data):
    global scan_points

    angle_increment_rad = data.angle_increment
    range_min = data.range_min
    range_max = data.range_max

    data_points = ([],[]) # (X,Y)
    # Form (x,y) with polar -> cartesian conversion
    for index, range in enumerate(data.ranges):
        # Discard smaller / greater than the corresponding range values for the sensor
        if range < range_min or range_max < range:
            continue
        theta = angle_increment_rad * index # float as default from int * float
        data_points[0].append(range * np.cos(theta))
        data_points[1].append(range * np.sin(theta))
    
    scan_points = np.asarray(data_points)


def callback_obstacles(data):
    global obstacles
    obstacles = data.obstacles


def callback_goal(data):
    global relative_goal
    relative_goal = (data.x, data.y)


def callback_control_output(data):
    global control_output
    control_output = (data.ux, data.uy)


def on_click(event):
    goal = Point()
    if event.inaxes:
        if event.button is mb.LEFT:
            goal.x = event.xdata
            goal.y = event.ydata
            goal.z = 0.0
            goal_pub.publish(goal)
        elif event.button is mb.RIGHT:
            goal_pub.publish(goal)


if __name__ == "__main__":
    rospy.init_node('ui_node')

    scan_sub = rospy.Subscriber(NAMESPACE+"/scan", LaserScan, callback_scan)
    goal_sub = rospy.Subscriber(NAMESPACE+"/relative_goal", Point, callback_goal)
    obstacle_sub = rospy.Subscriber(NAMESPACE+'/obstacles', ObstacleArray, callback_obstacles)
    control_output_sub = rospy.Subscriber(NAMESPACE+'/control_output', SIControlOutput, callback_control_output)
    ui = TurtlebotUI()

    plt.connect('button_press_event', on_click)

    while 1:
        try:
            ui.update_plot(DRAW_ESTIMATED_OBSTACLES)
            time.sleep(UPDATE_RATE)
        except TclError:
            goal_pub.publish(Point())
            break
    
