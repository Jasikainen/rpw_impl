# rpw_impl
Robotics Project Work - Lidar assisted collision avoidance of turtlebot3 burger

## First step is to clone the repository to catkin workspace

Go to following
> cd ~/catkin_ws/src

Clone the folder
> git clone git@github.com:Jasikainen/rpw_impl.git

## Next step is to do the following

Running catkin_make is needed here after getting the repository
> cd ~/catkin_ws/

> catkin_make

### Installing required packages
Install the ´scikit-learn´
> pip3 install -U scikit-learn

### How to perform scripts specified
Run for example script from "scripts/lidar_detection.py" to start the LIDAR node

First do the following to start turtlebot3 simulation
> export TURTLEBOT3_MODEL=burger

To start off experimenting launch premade gazebo world 
> roslaunch turtlebot3_gazebo turtlebot3_world.launch

Start Rviz visualizer
> roslaunch turtlebot3_gazebo turtlebot3_gazebo_rviz.launch

Run from the repository the specific script
> rosrun rpw_impl lidar_detection.py

The controller and the ui nodes can be run similarly
> rosrun rpw_impl controller.py

> rosrun rpw_impl ui.py

#### Using .launch files

##### Launch all nodes
> roslaunch rpw_impl ui_and_controller.launch

##### If you run this from the turtlebot, you should use
> roslaunch rpw_impl controller_detector.launch

and start the ui separately from the remote pc

##### Launching with optional arguments
Obstacle detection supports three different types of actions. All of them include location (x,y) of the detected obstacle and it's radius. All of algorithms, but the **first**,  use default radius ```r``` for the detected obstacles. First algorithm evaluates the radius based on the points within the cluster.

- ```obstactle_detection``` (e.g. How obstacles are fed to controller)
  - MULTIPLE_OBSTACLES: LIDAR points converted into clusters that are fed to controller as multiple obstacles
  - CLOSEST_LIDAR_POINT: Single closest LIDAR point (raw data)
  - CLOSEST_CLUSTER_POINTS: LIDAR points converted into cluster but only closest point of each cluster is used as an obstacle

Define the namespace that is used for Turtlebot 3 Burger

- ```topic_namespace``` (e.g. using physical turtlebot3 burger's namespace)


Values are given without ```<``` and ```>``` in the following
#
```sh
> roslaunch rpw_imp ui_and_controller.launch topic_namespace:=</name_of_namespace> obstactle_detection:=<MULTIPLE_OBSTACLES or CLOSEST_LIDAR_POINT or CLOSEST_CLUSTER_POINTS> 
```

```sh
> roslaunch rpw_imp controller.launch topic_namespace:=<name_of_namespace> obstactle_detection:=<MULTIPLE_OBSTACLES or CLOSEST_LIDAR_POINT or CLOSEST_CLUSTER_POINTS>
```

### Add new scripts and/or edit the existing

Feel free to edit the existing scripts inside "scripts/" folder and commit the changes to repository
