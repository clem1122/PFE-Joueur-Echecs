# !/usr/bin/env python3

from pyniryo import *

robot = NiryoRobot("10.10.10.10")

robot.calibrate_auto()
robot.update_tool()

robot.release_with_tool()
robot.move_pose(0.2, -0.1, 0.35, 0.0, 1.57, 0.0)
robot.move_pose(0.2, -0.1, 0.25, 0.0, 1.57, 0.0)

robot.grasp_with_tool()
robot.move_pose(0.2, -0.1, 0.35, 0.0, 1.57, 0.0)
robot.move_pose(0.2, 0.1, 0.35, 0.0, 1.57, 0.0)
robot.move_pose(0.2, 0.1, 0.25, 0.0, 1.57, 0.0)
robot.release_with_tool()

robot.move_to_home_pose()


robot.close_connection()



