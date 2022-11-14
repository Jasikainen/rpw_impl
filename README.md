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

Or to launch all nodes
> roslaunch rpw_impl ui_and_controller.launch

If you run this from the turtlebot, you should use
> roslaunch rpw_impl controller_detector.launch

and start the ui separately from the remote pc

### Add new scripts and/or edit the existing

Feel free to edit the existing scripts inside "scripts/" folder and commit the changes to repository
