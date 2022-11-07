import rospy
import time
from sensor_msgs.msg import LaserScan
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrow
from matplotlib.backend_bases import MouseButton as mb
from rpw_impl.msg import ObstacleData, ObstacleArray
from geometry_msgs.msg import Point
from _tkinter import TclError


TOPIC_PREFIX = ""
DRAW_ESTIMATED_OBSTACLES = True
UPDATE_RATE = 0.001

# Globals
scan_points = ([],[])
obstacles = []
relative_goal = (0.0,0.0) # From controller
goal_pub = rospy.Publisher("/new_goal", Point, queue_size=10)


class TurtlebotUI:
    def __init__(self,AxisLimits=[-3,3,-3,3]):
        plt.ion()
        self.al = AxisLimits
        self.fig = plt.figure(num="Turtlebot UI", facecolor="Black",figsize=[8.0,8.0])
        self.fig.set_tight_layout(True)
        self.ax = self.fig.gca()
        self.ax.set_aspect('equal')
        self.ax.set_facecolor("Black")
        self.ax.tick_params(colors="Black")
        self.ax.set_xlim(self.al[0],self.al[1])
        self.ax.set_ylim(self.al[2],self.al[3])
        self.ax.set_xlabel("X (m)")
        for spine in self.ax.spines.values():
            spine.set_edgecolor('gray')
        self.ax.set_ylabel("Y (m)", rotation="horizontal")
        self.ax.plot(0,0, color='gray', marker='o',markersize=20)
        self.ax.plot(0.2,0, color='gray', marker='>',markersize=6)
        self.scatter = self.ax.scatter(scan_points[0],scan_points[1],c="Red")
        self.objects = []

    def refresh(self):
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update_plot(self, plot_obstacles=False):
        if not self.ax:
            print("yay")
        self.clear_plot()

        # Plot lidar data points
        self.scatter = self.ax.scatter(scan_points[0],scan_points[1],s=1.0,c="Red")

        # Plot estimated obstacles
        if plot_obstacles:
            for obstacle in obstacles:
                circle = Circle(xy=obstacle.center, radius=obstacle.radius, color="Red",fill=True,alpha=0.1)
                self.ax.add_patch(circle)

        # Plot goal
        goal = FancyArrow(0,0,relative_goal[0],relative_goal[1],width=.03,
                            length_includes_head=True, color="Yellow", alpha=.25)
        self.ax.add_patch(goal)

        self.refresh()

    def clear_plot(self):
        [p.remove() for p in reversed(self.ax.patches)]
        self.scatter.remove()


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


def on_click(event):
    if event.button is mb.LEFT:
        if event.inaxes:
            goal = Point()
            goal.x = event.xdata
            goal.y = event.ydata
            goal.z = 0.0
            goal_pub.publish(goal)


if __name__ == "__main__":
    rospy.init_node('ui_node')

    scan_sub = rospy.Subscriber(TOPIC_PREFIX+"/scan", LaserScan, callback_scan)
    goal_sub = rospy.Subscriber("/relative_goal", Point, callback_goal)
    obstacle_sub = rospy.Subscriber('/obstacles', ObstacleArray, callback_obstacles)
    
    ui = TurtlebotUI()

    plt.connect('button_press_event', on_click)

    while 1:
        try:
            ui.update_plot(DRAW_ESTIMATED_OBSTACLES)
            time.sleep(UPDATE_RATE)
        except TclError:
            break
    
