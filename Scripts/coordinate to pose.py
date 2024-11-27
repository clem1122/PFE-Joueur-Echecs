# !/usr/bin/env python3

from pyniryo import *

# === Parameters ==

robot_ip = "192.168.148.1"
square_size = 0.032

robot = NiryoRobot(robot_ip)


# === Functions to play a move ===

def coord_to_robot(depart,arrivee):
    y1 = ord(depart[0]) - 96
    y2 = ord(arrivee[0]) - 96
    x1 = int(depart[1])
    x2 = int(arrivee[1])

    return x1,y1,x2,y2


def coo_to_pose(nb_row, nb_column, z):
    # z=0,1,1.5,6 (low,middle,high, above)
    #Departure from a1
    return a1_pose.copy_with_offsets(x_offset = square_size * (nb_row -1), y_offset = -square_size * (nb_column -1) , z_offset = 0.02 * z)


def go_take_move_release(x1,y1,x2,y2,z_grasp_piece):
    #x1,y1,x2,y2 = 4,1,4,8 #coord_to_robot(depart,arrivee)
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


# === Calibration ===

#origin_pose = robot.get_pose_saved("H8_Coord")
a1_pose = PoseObject(x=0.204494, y = 0.107565, z = 0.145907, roll = -0.501, pitch = 1.511, yaw = 2.041)

#robot.move_pose(origin_pose)
#go_take_move_release("f2","f3",0)
#robot.move_pose(a1_pose.copy_with_offsets(x_offset = 0, y_offset = -0.1 , z_offset = 0.02 * 6))
#Voir les tajectory pour avoir mouvement plus fluide

#observation_pose = PoseObject(x=0.0, y=-0.19, z=0.35, roll=-3.13, pitch=1.40, yaw = 1.56)


#robot.calibrate_auto()
go_take_move_release(4,1,4,7,0)
#robot.update_tool()
#robot.clear_collision_detected()

#robot.move_pose(0.16, 0.136, 0.15, 3., 0.65, 0)

#robot.move_pose(origin_pose.copy_with_offsets(x_offset = 0.04 * 2, y_offset = -0.04 * 4, z_offset = 0.0))
#robot.grasp_with_tool()






robot.close_connection()

