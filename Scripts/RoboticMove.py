import PChess as pc

class Space:
	def __init__(self):
		self.chessboard = self.generate()
		
	def generate(self):
		square_size = 5
		columns = "abcdefgh"
		rows = range(1, 9)
		chessboard = {}
		for i, column in enumerate(columns):
			for j, row in enumerate(rows):
				square_name = f"{column}{row}"
				x = i * square_size
				y = j * square_size
				chessboard[square_name] = (x, y)
				
		for i in range(1, 9):
			square_name = "v" + str(i)
			chessboard[square_name] = (-10, i * square_size)
		
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
