class Space:
	def __init__(self):
		self.a1_pose = [0.150, 0.136, 0.097]
		self.square_size = 0.04
		self.chessboard = self.generate()
		
		
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
				
		for i in range(0, 4):
			for j in range(1,5):
				b_square_name = "v" + str((i*4)+j)
				w_square_name = "V" + str((i*4)+j)

				chessboard[b_square_name] = [self.a1_pose[0] + i*self.square_size, self.a1_pose[1] + j * self.square_size, self.a1_pose[2]]
				chessboard[w_square_name] = [self.a1_pose[0] + i*self.square_size, self.a1_pose[1] + (j-12) * self.square_size, self.a1_pose[2]]
					
				
		return chessboard

class Height:
	def __init__(self):
		self.LOW = 0
		self.MID = 0.02
		self.HIGH = 0.03
		self.ABOVE = 0.12
		
space = Space()
height = Height()
