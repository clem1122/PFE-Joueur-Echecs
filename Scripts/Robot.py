from pyniryo import *
from Space import space, height
from RoboticMove import RoboticMove, TestRoboticMove, create_complex_robotic_move


class Robot:

	def __init__(self):
		self.ip = "192.168.73.1"
		self.niryo = NiryoRobot(self.ip)
		self.niryo.calibrate_auto()
		self.niryo.set_arm_max_velocity(100)

	def authorize_move(self,robotic_move):
		print(robotic_move)
		validation = input("\n Le mouvement doit-il être effectué ? [y/n] \n")
		while validation != "y" or validation != "n":
			validation = input("\n Le mouvement doit-il être effectué ? [y/n] \n")

		author_bool = (validation=="y")

		return author_bool
	
	def move_to_obs_pose(self):
		self.niryo.move_joints(space.observation_joints)	
		
	def move_to_square(self, square):
		if len(square) != 2:
			raise Exception("Uncorrect Move argument")
		list_coord = space.chessboard[square] + [2.36, 1.57, -3.14]
		self.niryo.move_pose(self.get_pose(list_coord, height.HIGH))
	

	def get_pose(self, coord_list, h):
		return PoseObject(coord_list[0], coord_list[1], coord_list[2]+h, coord_list[3], coord_list[4], coord_list[5])

	def play_move(self, board, PChess_move):
		
		isComplex = (PChess_move.isCapture() + PChess_move.isPromoting() + PChess_move.isCastling() + PChess_move.isEnPassant()) > 0
		if not isComplex :
			rob_move = RoboticMove(PChess_move.start(),PChess_move.end(),PChess_move.moving_piece())
			if self.authorize_move(rob_move):
				self.execute_move(rob_move)
			
		else :
			complex_move_list = create_complex_robotic_move(board ,PChess_move)
			
			for robotic_move in complex_move_list :
				if self.authorize_move(robotic_move):
					self.execute_move(robotic_move)
			
	def play_test_move(self, PChess_move, h):
		isComplex = (PChess_move.isCapture() + PChess_move.isPromoting() + PChess_move.isCastling() + PChess_move.isEnPassant()) > 0
		
		if not isComplex :
			self.execute_move(TestRoboticMove(PChess_move, h))
		
			
	
	def execute_move(self, robotic_move):
		square1 = robotic_move.take_pose
		square2 = robotic_move.drop_pose
		

		pose1 = self.get_pose(square1, height.ABOVE).to_list()
		pose2 = self.get_pose(square1, robotic_move.piece_height).to_list()
		pose3 = self.get_pose(square1, height.ABOVE).to_list()
		pose4 = self.get_pose(square2, height.ABOVE).to_list()
		pose5 = self.get_pose(square2, robotic_move.piece_height).to_list()
		pose6 = self.get_pose(square2, height.ABOVE).to_list()

		traj_to_piece  = [pose1, pose2]
		traj_to_square = [pose3, pose4, pose5]


		traj_to_home = [pose6]

		
		self.niryo.execute_trajectory_from_poses_and_joints(traj_to_piece)
		self.niryo.close_gripper()
		self.niryo.execute_trajectory_from_poses_and_joints(traj_to_square, dist_smoothing = 0.5)
		self.niryo.open_gripper()
		self.niryo.execute_trajectory_from_poses_and_joints(traj_to_home)

		if robotic_move.isEndMove:
			self.niryo.move_joints(space.observation_joints)

	def close(self):
		self.niryo.close_connection()

