class Space:
	def __init__(self):
		self.a1_pose = [0.146, 0.129, 0.108]
		self.square_size = 0.037
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
		self.MID = 0.025
		self.HIGH = 0.035
		self.ABOVE = 0.125
		self.pieces_height = self.generate_height_dictionary()
		
	def generate_height_dictionnary(self):
		
		height_dictionary = {}
		pieces_list = ['r','n','b','q','k','p']
		height_list = ['MID','MID','MID','HIGH','HIGH','LOW']
		
		for i,piece in pieces_list :
			height_dictionary[piece] = height_list[i]
			height_dictionary[upper(piece)] = height_list[i]
	
		return height_dictionary
space = Space()
height = Height()

