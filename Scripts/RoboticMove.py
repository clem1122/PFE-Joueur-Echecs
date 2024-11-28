import PChess as pc


class Space:
	def __init__(self):
		self.chessboard = self.generate()
		self.a1_pose = [0.150, 0.136, 0.100]
		
	def generate(self):
		square_size = 4
		columns = "abcdefgh"
		rows = range(1, 9)
		chessboard = {}
		for i, column in enumerate(columns):
			for j, row in enumerate(rows):
				square_name = f"{column}{row}"
				x = a1_pose[0] + i * square_size
				y = a1_pose[1] -j * square_size
				z = a1_pose[2]
				chessboard[square_name] = (x, y, z)
				
		for i in range(0, 4):
		    for j in range(1,5):
			b_square_name = "v" + str((i*4)+j)
			w_square_name = "V" + str((i*4)+j)

			chessboard[b_square_name] = (i*square_size, -j * square_size,self.a1_pose[2])
			chessboard[w_square_name] = (i*square_size, (12-j) * square_size,self.a1_pose[2])
					
		
		return chessboard



class RoboticMove:
	def __init__(self, A, B):
		s = Space();
		self.take_coord = s.chessboard[A];
		self.drop_coord = s.chessboard[B];
	def __str__(self):
		return "Robotic move from " + str(self.take_coord) + " to " + str(self.drop_coord)


def create_robotic_move(move):
	return RoboticMove(move.start(), move.end())
		



def create_complex_robotic_move(move):
	A = move.start()
	B = move.end()
	if(move.isCapture() and move.isPromotion()):
		raise Exception("Promotin + Capture")
		
	if(move.isCapture()):
		V = valhalla(A)
		return [RoboticMove(B, V), RoboticMove(A, B)]
	elif(move.isCastling()):
		C = "a1"
		D = "a2"
		return [RoboticMove(A, B), RoboticMove(C, D)]
	elif(move.isPromotion()):
		F = valhalla(A)
		V = valhalla(A)
		return [RoboticMove(A, V), RoboticMove(F, B)]
	elif(move.isEnPassant()):
		V = valhalla(A)
		E = "a4";
		return [RoboticMove(A, B), RoboticMove(E, V)]
	else:
		raise Exception("Move is not complex")

def valhalla(A):
	return "v1"
