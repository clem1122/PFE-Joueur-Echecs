# !/usr/bin/env python3

from pyniryo import *


robot_ip = "192.168.10.1"

robot = NiryoRobot(robot_ip)

origin_pose = PoseObject(x=0.180, y = 0.136, z = 0.100, roll = 3.09, pitch = 1.43, yaw = -3.10)


def coo_to_pose(nb_row, nb_column, z):
    # z=0,1,1.5,6 (low,middle,high, above)
    #Departure from H8
    return origin_pose.copy_with_offsets(x_offset = 0.04 * (nb_row -1), y_offset = -0.04 * (nb_column -1) , z_offset = 0.02 * z)

def go_take_move_release(x1,y1,x2,y2,z_grasp_piece):
    robot.move_to_home_pose()
    robot.move_pose(coo_to_pose(x1, y1, 4))
    robot.move_pose(coo_to_pose(x1, y1, z_grasp_piece))
    robot.close_gripper(speed = 800)
    robot.move_pose(coo_to_pose(x1, y1, 6))
    robot.move_pose(coo_to_pose(x2, y2, 6))
    robot.move_pose(coo_to_pose(x2, y2, z_grasp_piece))
    robot.open_gripper(speed = 800)
    robot.move_pose(coo_to_pose(x2, y2, 6))
    robot.move_to_home_pose()

#Voir les tajectory pour avoir mouvement plus fluide

#observation_pose = PoseObject(x=0.0, y=-0.19, z=0.35, roll=-3.13, pitch=1.40, yaw = 1.56)


robot.calibrate_auto()
go_take_move_release(2,5,4,5,1.5)
#robot.update_tool()
#robot.clear_collision_detected()

#robot.move_pose(0.16, 0.136, 0.15, 3., 0.65, 0)

#robot.move_pose(origin_pose.copy_with_offsets(x_offset = 0.04 * 2, y_offset = -0.04 * 4, z_offset = 0.0))
#robot.grasp_with_tool()






robot.close_connection()
