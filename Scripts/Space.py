import PChess as pc
b = pc.Board()

class Space:
	def __init__(self):
		self.a1_pose = [0.146, 0.129, 0.102]
		self.square_size = 0.04
		self.valhalla_square_size = 0.04
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
				
		valhalla_offset_y = 0.7  #square
		valhalla_offset_z = 0.00 #m
		for i in range(0, 4):
			for j in range(1,6):
				b_square_name = "v" + b.to_base((i*5)+j, 20)
				w_square_name = "V" + b.to_base((i*5)+j, 20)
				

				chessboard[b_square_name] = [self.a1_pose[0] + i*self.valhalla_square_size, self.a1_pose[1] + (j + valhalla_offset_y) * self.valhalla_square_size, self.a1_pose[2]-valhalla_offset_z]
				chessboard[w_square_name] = [self.a1_pose[0] + i*self.valhalla_square_size, self.a1_pose[1] + (j-13 - valhalla_offset_y) * self.valhalla_square_size, self.a1_pose[2] - valhalla_offset_z]
					
				
		return chessboard


		
class Height:
	def __init__(self):
		self.LOW = 0
		self.MID = 0.025
		self.HIGH = 0.035
		self.ABOVE = 0.175
		self.pieces_height = self.generate_height_dictionary()
		
	def generate_height_dictionary(self):
		
		height_dictionary = {}
		pieces_list = ['r','n','b','q','k','p']
		height_list = [self.LOW,self.MID,self.MID,self.HIGH,self.HIGH,self.LOW]
		
		for i,piece in enumerate(pieces_list) :
			height_dictionary[piece] = height_list[i]
			height_dictionary[piece.upper()] = height_list[i]
	
		return height_dictionary
space = Space()
height = Height()

