import Scripts.PChess as pc
b = pc.Board()

class Space:
	def __init__(self):
		self.a1_pose = [0.140, 0.1379, 0.124] #0.136, 0.129, 0.114
		self.square_size = 0.04
		self.chessboard = self.generate()
		self.observation_joints = [0, 0.365, -0.393, -0.028, -1.706, -0.029] #0.022, 0.327, -0.392, -0.026, -1.651, -0.011
		self.w_valhalla_joints = [-0.888, -0.017, -0.245, 0, -1.5, -0.064] #0.888, -0.017, -0.245, 0, -1.5, -0.064
		self.b_valhalla_joints = [0.888, -0.017, -0.245, 0, -1.5, -0.064] #-0.888, -0.017, -0.245, 0, -1.5, -0.064
		
		
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
				
		valhalla_offset_y = 1  #square
		valhalla_offset_x = 0  #square
		valhalla_offset_z = 0.005 #m
		for i in range(0, 6):
			for j in range(1,5):
				b_square_name = "v" + b.to_base((i*4)+j, 20)
				w_square_name = "V" + b.to_base((i*4)+j, 20)
				

				chessboard[b_square_name] = [self.a1_pose[0] + (i + valhalla_offset_x)*self.square_size, self.a1_pose[1] + (j + valhalla_offset_y) * self.square_size, self.a1_pose[2]-valhalla_offset_z]
				chessboard[w_square_name] = [self.a1_pose[0] + (i + valhalla_offset_x)*self.square_size, self.a1_pose[1] + (j-12 - valhalla_offset_y) * self.square_size, self.a1_pose[2] - valhalla_offset_z]
					
				
		return chessboard


		
class Height:
	def __init__(self):
		self.LOW = 0.01
		self.MID = 0.02
		self.HIGH = 0.029
		self.POINT = 0.040
		self.ABOVE = 0.15
		
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

