# !/usr/bin/env python3

from pyniryo import *
from cv2 import imwrite
import numpy as np

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

#origin_pose = PoseObject(x=0.150, y = 0.136, z = 0.100, roll = 2.36, pitch = 1.57, yaw = -3.14)

observation_pose = [0.022, 0.327, -0.392, -0.026, -1.651, -0.011]
squareSize = 0.04


def coo_to_pose(nb_column, nb_row, z = 0):
    # z=0,1,1.5,6 (low,middle,high, above)
    #Departure from a1
    return a1_pose.copy_with_offsets(x_offset = square_size * (nb_row -1), y_offset = -square_size * (nb_column -1) , z_offset = 0.02 * z)


def go_take_move_release(x1,y1,x2,y2,z_grasp_piece):
    #x1,y1,x2,y2 = 4,1,4,8 #coord_to_robot(depart,arrivee)
    robot.move_to_home_pose()
    robot.move_pose(coo_to_pose(x1, y1, 4))
    robot.move_pose(coo_to_pose(x1, y1, z_grasp_piece))
    #Departure from H8

    return origin_pose.copy_with_offsets(x_offset = 0.04 * (nb_row -1), y_offset = -0.04 * (nb_column -1) , z_offset = 0.02 * z)

def go_take_move_release(move,z_grasp_piece, isEndMove = True):
    x1, x2 = ord(move[0]) - 96, ord(move[2]) - 96
    y1, y2 = int(move[1]), int(move[3])
    
    pose1 = coo_to_pose(x1, y1, 4).to_list()
    pose2 = coo_to_pose(x1, y1, z_grasp_piece).to_list()
    pose3 = coo_to_pose(x1, y1, 6).to_list()
    pose4 = coo_to_pose(x2, y2, 6).to_list()
    pose5 = coo_to_pose(x2, y2, z_grasp_piece).to_list()
    pose6 = coo_to_pose(x2, y2, 6).to_list()

    traj_to_piece  = [pose1, pose2]
    traj_to_square = [pose3, pose4, pose5]

    traj_to_home =   [pose6]
    if isEndMove:
        traj_to_home.append(observation_pose.to_list())

    robot.execute_trajectory_from_poses(traj_to_piece)
    robot.close_gripper(speed = 800)
    robot.execute_trajectory_from_poses(traj_to_square, dist_smoothing = 0.8)
    robot.open_gripper(speed = 800)
    robot.execute_trajectory_from_poses(traj_to_home)




# === Calibration ===

#origin_pose = robot.get_pose_saved("H8_Coord")
a1_pose = PoseObject(x=0.204494, y = 0.107565, z = 0.145907, roll = -0.501, pitch = 1.511, yaw = 2.041)

#robot.move_pose(origin_pose)
#go_take_move_release("f2","f3",0)
#robot.move_pose(a1_pose.copy_with_offsets(x_offset = 0, y_offset = -0.1 , z_offset = 0.02 * 6))
#Voir les tajectory pour avoir mouvement plus fluide

#observation_pose = PoseObject(x=0.0, y=-0.19, z=0.35, roll=-3.13, pitch=1.40, yaw = 1.56)



robot.calibrate_auto()
mtx,dist = robot.get_camera_intrinsics()
#go_take_move_release("d5e7", 1)

robot.move_joints(observation_pose)
raw_image = robot.get_img_compressed()
image_uncompressed = uncompress_image(raw_image)
img_undistort = undistort_image(image_uncompressed,mtx,dist)
#go_take_move_release("e7d5",1)
imwrite("D:/Documents/Python Scripts/PFE-Joueur-Echecs/Photos/photo4.png", img_undistort)
image_functions.show_img("windows", img_undistort,wait_ms=10000)



robot.close_connection()

