from pyniryo import *
from Space import space, height


class Robot:

	def __init__(self):

		self.ip = "192.168.94.1"

		self.niryo = NiryoRobot(robot_ip)
		
	def move_to_square(square):
		if len(square) != 2:
			raise Exception("Uncorrect Move argument")
		
		self.niryo.move_pose(get_pose(square))
	

	def get_pose(square, height = height.LOW):
		orientation = [2.36, 1.57, -3.14]
		poseList = space.chessboard[square] + orientation
		return PoseObject(poseList[0], poseList[1], poseList[2] + height, poseList[3], poseList[4], poseList[5])

	def play_move(PChess_move):
		pass
	
	
	def execute_move(robotic_move):
		square1 = robotic_move.take_coord
		square2 = robotic_move.drop_coord
		

		pose1 = get_pose(square1, height.ABOVE).to_list()
		pose2 = get_pose(square1, robotic_move.piece_height).to_list()
		pose3 = get_pose(square1, height.ABOVE).to_list()
		pose4 = get_pose(square2, height.ABOVE).to_list()
		pose5 = get_pose(square2, robotic_move.piece_height).to_list()
		pose6 = get_pose(square2, height.ABOVE).to_list()

		traj_to_piece  = [pose1, pose2]
		traj_to_square = [pose3, pose4, pose5]

		traj_to_home = [pose6]


		self.niryo.execute_trajectory_from_poses_and_joints(traj_to_piece)
		self.niryo.close_gripper()
		self.niryo.execute_trajectory_from_poses_and_joints(traj_to_square, dist_smoothing = 0.8)
		self.niryo.open_gripper()
		self.niryo.execute_trajectory_from_poses_and_joints(traj_to_home)

		if robotic_move.isEndMove:
			self.niryo.move_joints(observation_joints)




