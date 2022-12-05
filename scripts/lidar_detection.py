#!/usr/bin/env python
import rospy
import argparse
from sensor_msgs.msg import LaserScan
import tf, math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
from sklearn.cluster import DBSCAN # LINUX pip install: pip3 install -U scikit-learn
from sklearn import metrics
import seaborn as sns
from collections import Counter
from rpw_impl.msg import ObstacleData, ObstacleArray
from enum import Enum


class DetectObstacles(Enum):
    MULTIPLE_OBSTACLES = 1
    CLOSEST_LIDAR_POINT = 2
    CLOSEST_CLUSTER_POINTS = 3

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(str):
        try:
            return DetectObstacles[str.upper()]
        except KeyError as error:
            raise ValueError()


parser = argparse.ArgumentParser()
parser.add_argument("--namespace",
            help="prepend all topics with this namespace",
            default=rospy.get_namespace().rstrip("/"))

parser.add_argument("--disable_output",
            action='store_true',
            help="disables all output")

parser.add_argument("--detect_obstacles",
            type=DetectObstacles.from_string,
            choices=list(DetectObstacles),
            help="choose how obstacles are handled based on LIDAR data")         

args = parser.parse_known_args()
NAMESPACE = args[0].namespace.rstrip("/")

ENABLE_OUTPUT = not args[0].disable_output
OBSTACLE_DETECTION = args[0].detect_obstacles
SAFE_MARGIN = 0.3 # Same values as in controller.py
MAX_RADIUS = 0.5
DEFAULT_OBJECT_LABEL = 1 # Any positive number
DEFAULT_OBJECT_RADIUS = 0.05

# Use as global variables
labels_ext      = 0
data_points_ext = 0
objects_ext     = {}
obstacle_pub = rospy.Publisher(NAMESPACE+"/obstacles", ObstacleArray, queue_size=10)


def callback(data):
    # Global definitions if these are needed AND changed during execution
    global labels_ext, data_points_ext

    # Get the values from the data object received
    angle_increment_rad = data.angle_increment
    angle_min = data.angle_min
    angle_max = data.angle_max
    # Values between range limits should be considered
    range_min = data.range_min
    range_max = data.range_max

    # Convert data to numpy array to insert into DBSCAN clustering algorithm
    data_points = []
    # Form (x,y) with polar -> cartesian conversion
    for index, range in enumerate(data.ranges):
        # Discard smaller / greater than the corresponding range values for the sensor
        if range < range_min or range_max < range:
            continue
        theta = angle_increment_rad * index # float as default from int * float
        data_points.append([range * math.cos(theta), range * math.sin(theta)])
    data_points_ext = np.asarray(data_points)

    # No further actions required if zero points detected
    if data_points_ext.size == 0:
        update_objects({})
        return

    db = DBSCAN(eps=0.2, min_samples=5).fit(data_points_ext)
    labels_ext = db.labels_ # Non unique values (e.g. all points)

    if ENABLE_OUTPUT:
        # Calculate the amount of clusters found excluding noise!
        n_clusters_ = len(set(labels_ext)) - (1 if -1 in labels_ext else 0)
        # How many were labeled as -1 e.g. noise
        n_noise_ = list(labels_ext).count(-1)
        print("Clusters found: %d" % n_clusters_)
        print("Points classified as noise: %d" % n_noise_)
        if n_clusters_ >= 2:
            print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(data_points_ext, labels_ext))

    clusters = points_to_clusters(data_points)
    update_objects(clusters)


def points_to_clusters(data_points):
    clusters = {}

    if OBSTACLE_DETECTION is DetectObstacles.CLOSEST_LIDAR_POINT:
        # Computes row vise L2-norm for all of (x,y)-coordinate pairs (LIDAR data)
        pnt_distances = np.sum(np.abs(data_points)**2,axis=-1)**(1./2)
        closest_pnt_index = np.argmin(pnt_distances)
        clusters[DEFAULT_OBJECT_LABEL] = ([data_points[closest_pnt_index][0]],[data_points[closest_pnt_index][1]])
        return clusters

    # Form clusters based on the labels points have been assigned
    for pnt, label in zip(data_points,labels_ext):
        if label in clusters:
            clusters[label][0].append(pnt[0])
            clusters[label][1].append(pnt[1])
        else:
            clusters[label] = ([pnt[0]],[pnt[1]])

    if OBSTACLE_DETECTION is DetectObstacles.MULTIPLE_OBSTACLES:
        return clusters

    # Overwrite existing clusters by changing "multiple points per cluster" into a "single closest point per cluster"
    if OBSTACLE_DETECTION is DetectObstacles.CLOSEST_CLUSTER_POINTS:
        
        clusters_overwritten = {}
        for label, points_in_cluster in clusters.items():
            pnt_distances = np.linalg.norm(points_in_cluster, axis=0)
            closest_pnt_index = np.argmin(pnt_distances)
            clusters_overwritten[label] = ([points_in_cluster[0][closest_pnt_index]],[points_in_cluster[1][closest_pnt_index]])
        return clusters_overwritten


def fit_circle(x, y):  
    if len(x) == 1:
        return [x[0], y[0]], 0.05
    if len(x) == 2:
        xc = np.mean(x)
        yc = np.mean(y)
        r = np.linalg.norm([x[1]-x[0], y[1]-y[0]])/2
        return [xc, yc], r
    A = np.array([x, y, np.ones(len(x))]).T
    b = np.array(x)**2 + np.array(y)**2
    c = np.linalg.lstsq(A,b,rcond=None)[0]
    xc = c[0]/2
    yc = c[1]/2
    r = np.sqrt(c[2] + xc**2 + yc**2)
    return [xc, yc], r


def better_estimate_for_large_clusters(cluster, radius):
    pnt_distances = np.linalg.norm(cluster, axis=0)
    closest_pnt_i = np.argmin(pnt_distances)
    closest_pnt = [cluster[0][closest_pnt_i], cluster[1][closest_pnt_i]]
    closest_pnt_unit = closest_pnt/pnt_distances[closest_pnt_i]
    center = closest_pnt + closest_pnt_unit*(radius-SAFE_MARGIN)
    return center


def update_objects(clusters):
    global objects_ext
    objects_ext = {}
    
    for label, cluster in clusters.items():
        if OBSTACLE_DETECTION is DetectObstacles.MULTIPLE_OBSTACLES:
            # From noise choose only the closest point
            if label == -1:
                pnt_distances = np.linalg.norm(cluster, axis=0)
                closest_pnt_i = np.argmin(pnt_distances)
                closest_pnt = [cluster[0][closest_pnt_i], cluster[1][closest_pnt_i]]
                objects_ext[label] = (closest_pnt, DEFAULT_OBJECT_RADIUS, pnt_distances[closest_pnt_i]-DEFAULT_OBJECT_RADIUS)
                continue

            # Estimate a circle based on the points in a cluster
            center, radius = fit_circle(cluster[0], cluster[1])
            if radius > MAX_RADIUS:
                center = better_estimate_for_large_clusters(cluster, radius)

            center_distance = np.linalg.norm(center)
            distance_from_turtlebot = center_distance - radius
            objects_ext[label] = (center, radius, distance_from_turtlebot)

        elif OBSTACLE_DETECTION is DetectObstacles.CLOSEST_LIDAR_POINT or OBSTACLE_DETECTION is DetectObstacles.CLOSEST_CLUSTER_POINTS:
            closest_point = [cluster[0][0], cluster[1][0]] # Second index 0 as single point should be within a cluster
            objects_ext[label] = (closest_point, DEFAULT_OBJECT_RADIUS, np.linalg.norm(cluster, axis=0)-DEFAULT_OBJECT_RADIUS)
            
            # Since only a one cluster should be found for "single datapoint obstacle detection" we can stop here
            if OBSTACLE_DETECTION is DetectObstacles.CLOSEST_LIDAR_POINT:
                break

    publish_obstacles()


def publish_obstacles():
    obstacles = []
    for label, obj in objects_ext.items():
        obstacle = ObstacleData()
        obstacle.name = str(label)
        obstacle.center = obj[0]
        obstacle.radius = obj[1]
        obstacle.distance = obj[2]
        obstacles.append(obstacle)
    
    obstacle_pub.publish(obstacles)


def detect_incoming_lidar_data(create_plot=False):
    rospy.init_node('lidar_node', anonymous=True)
    scan = rospy.Subscriber(NAMESPACE+"/scan", LaserScan, callback) # frame_id: /base_scan as in real tb3 it's tb3_1/base_scan (?)
    if ENABLE_OUTPUT:
        rospy.loginfo("Started the node")

    if create_plot:
        plot = ObstaclePlot()
        ani = FuncAnimation(plot.fig, plot.update_plot)
        plt.show(block=True) 

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


class ObstaclePlot:
    def __init__(self,AxisLimits=[-3,3,-3,3]):
        self.al = AxisLimits
        self.fig, self.ax = plt.subplots()
        self.ax.set_aspect('equal')
        self.ax.set_xlim(self.al[0],self.al[1])
        self.ax.set_ylim(self.al[2],self.al[3])
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y", rotation="horizontal")
        self.ax.plot(0,0, color='gray', marker='o',markersize=10)
        self.ax.plot(0.2,0, color='gray', marker='>',markersize=4)


    def update_plot(self,frame):
        if len(objects_ext) == 0:
            return
        self.clear_patches()

        for lbl, object in sorted(objects_ext.items(),key=lambda d: d[1][2]):
            center = object[0]
            r = object[1]
            # distance = object[2]
            obj = Circle(xy=center, radius=r, color='gray', fill=True, label=lbl)
            safe_margin = Circle(xy=center, radius=r+SAFE_MARGIN, color='red', fill=False)
            self.ax.add_patch(obj)
            self.ax.add_patch(safe_margin)

    def clear_patches(self):
        [p.remove() for p in reversed(self.ax.patches)]


# Main script is 
if __name__ == '__main__':
    detect_incoming_lidar_data(ENABLE_OUTPUT)
    if ENABLE_OUTPUT:
        # Create scatter plot - note that cluster category -1 == noise
        plot = sns.scatterplot(data=data_points_ext, x=data_points_ext[:, 0], y=data_points_ext[:, 1], 
            hue=labels_ext, legend="full", palette="deep")

        sns.move_legend(plot, "upper left", bbox_to_anchor=(1.0, 1.0), title='Clusters')
        n_clusters = len(set(labels_ext)) - (1 if -1 in labels_ext else 0)

        # Display the objects as circles
        for lbl, object in objects_ext.items():
            obj = Circle(xy=object[0], radius=object[1], color='red', fill=False, label=lbl)
            plot.add_patch(obj)

        plt.title(f"Estimated number of clusters: {n_clusters}")
        plt.gca().set_aspect('equal')
        plt.show()

    
    plt.show()