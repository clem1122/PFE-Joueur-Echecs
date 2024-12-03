class Space:
	def __init__(self):
		self.a1_pose = [0.146, 0.129, 0.110]
		self.square_size = 0.037
		self.chessboard = self.generate()
		self.observation_joints = [0.022, 0.327, -0.392, -0.026, -1.651, -0.011]
		
		
	def generate(self):
		columns = "abcdefgh"
		rows = range(1, 9)
		chessboard = {}
		for i, column in enumerate(columns):
			for j, row in enumerate(rows):
				square_name = f"{column}{row}"
				y = self.a1_pose[1] - i * self.square_size
				x = self.a1_pose[0] + j * self.square_size
				z = self.a1_pose[2]
				chessboard[square_name] = [x, y, z]
				
		valhalla_offset_y = 1.3  #square
		valhalla_offset_z = 0.01 #cm
		for i in range(0, 4):
			for j in range(1,5):
				b_square_name = "v" + str((i*4)+j)
				w_square_name = "V" + str((i*4)+j)
				

				chessboard[b_square_name] = [self.a1_pose[0] + i*self.square_size, self.a1_pose[1] + (j + valhalla_offset_y) * self.square_size, self.a1_pose[2]-valhalla_offset_z]
				chessboard[w_square_name] = [self.a1_pose[0] + i*self.square_size, self.a1_pose[1] + (j-12 - valhalla_offset_y) * self.square_size, self.a1_pose[2] - valhalla_offset_z]
					
				
		return chessboard

class Height:
	def __init__(self):
		self.LOW = 0
		self.MID = 0.025
		self.HIGH = 0.035
		self.ABOVE = 0.125
		
space = Space()
height = Height()
