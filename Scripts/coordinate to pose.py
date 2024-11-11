# !/usr/bin/env python3

from pyniryo import *


origin_pose = PoseObject(x=0.093, y = -0.173, z = 0.110, roll = 2.23, pitch = 1.38, yaw = 0.68)

def coo_to_pose(x, y):
    return origin_pose.copy_with_offsets(x_offset = -0.05 * x, y_offset = -0.05 * y)

robot_ip = "10.10.10.10"

observation_pose = PoseObject(x=0.0, y=-0.19, z=0.35, roll=-3.13, pitch=1.40, yaw = 1.56)

origin_pose = PoseObject(x=0.093, y = -0.173, z = 0.110, roll = 2.23, pitch = 1.38, yaw = 0.68)

robot = NiryoRobot(robot_ip)
robot.calibrate_auto()
robot.update_tool()

robot.move_pose(coo_to_pose(0, 0))


robot.say("Bonjour tout le monde", 1)

robot.close_connection()
