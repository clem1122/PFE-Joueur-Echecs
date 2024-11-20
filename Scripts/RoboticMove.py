import PChess as pc

class Space:
	def __init__(self):
		self.chessboard = {
			"e2" : (190,200), 
			"e4" : (180,200)
		}

class RoboticMove:
	def __init__(self, A, B):
		s = Space();
		self.take_coord = s.chessboard[A];
		self.drop_coord = s.chessboard[B];
	def __str__(self):
		return "Robotic move from " + str(self.take_coord) + " to " + str(self.drop_coord)


def create_robotic_move(move):
	return RoboticMove(move.start(), move.end())
		



def complex_robotic_move_conversion(move):
	if(move.isCapture()):
		pass
	elif(move.isCastling()):
		pass
	elif(move.isPromotion()):
		pass
	elif(move.isEnPassant()):
		pass
	else:
		raise Exception("Move is not complex")
	
