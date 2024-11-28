from pyniryo import *
from Space import space, height

# === Parameters ==

robot_ip = "192.168.148.1"
square_size = 0.04

robot = NiryoRobot(robot_ip)
# === Functions to play a move ===



def move_to_square(square):
	if len(square) != 2:
		raise Exception("Uncorrect Move argument")
	
	
	robot.move_pose(get_pose(square))
	
def coord_to_robot(depart,arrivee):
    y1 = ord(depart[0]) - 96
    y2 = ord(arrivee[0]) - 96
    x1 = int(depart[1])
    x2 = int(arrivee[1])

    return x1,y1,x2,y2


#a1_pose = PoseObject(x=0.150, y = 0.136, z = 0.100, roll = 2.36, pitch = 1.57, yaw = -3.14)
observation_joints = [0.022, 0.327, -0.392, -0.026, -1.651, -0.011]
#observation_pose = PoseObject(x=179.163, y=2.692, z=347.626, roll=-2.974, pitch=1.425, yaw =-2.940)

def get_pose(square, height = height.LOW):
	orientation = [2.36, 1.57, -3.14]
	poseList = space.chessboard[square] + orientation
	return PoseObject(poseList[0], poseList[1], poseList[2] + height, poseList[3], poseList[4], poseList[5])

def execute_move(move,piece_height, isEndMove = True):
	if len(move) != 4:
		raise Exception("Uncorrect move lenght")

	square1 = move[0:2]
	square2 = move[2:4]

	pose1 = get_pose(square1, height.ABOVE).to_list()
	pose2 = get_pose(square1, piece_height).to_list()
	pose3 = get_pose(square1, height.ABOVE).to_list()
	pose4 = get_pose(square2, height.ABOVE).to_list()
	pose5 = get_pose(square2, piece_height).to_list()
	pose6 = get_pose(square2, height.ABOVE).to_list()

	traj_to_piece  = [pose1, pose2]
	traj_to_square = [pose3, pose4, pose5]

	traj_to_home = [pose6]


	robot.execute_trajectory_from_poses_and_joints(traj_to_piece)
	robot.close_gripper()
	robot.execute_trajectory_from_poses_and_joints(traj_to_square, dist_smoothing = 0.8)
	robot.open_gripper()
	robot.execute_trajectory_from_poses_and_joints(traj_to_home)

	if isEndMove:
		robot.move_joints(observation_joints)




