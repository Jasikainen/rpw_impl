# rpw_impl
Robotics Project Work - Lidar assisted collision avoidance of turtlebot3 burger

## First step is to close the repository to catkin workspace

Go to following
> cd ~/catkin_ws/src

Clone the folder as "mepo700" and go into it and remove .git/ folder
> git clone https://github.com/Jasikainen/rpw_impl.git

## Next step is to do the following

Running catkin_make is needed here after getting the repository
> cd ~/catkin_ws/

> catkin_make

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

To move the turtlebot3 burger if necessary (real world turtlebot3 will not rely on these specific inputs)
> roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch
### Add new scripts and/or edit the existing

Feel free to edit the existing scripts inside "scripts/" folder and commit the changes to repository

### TODO

The is a need to make lidar_detection.py script to listen to "tb3_1/base_scan" 