import PChess as pc
from Space import space, height


		
class RoboticMove:
	def __init__(self,PChess_move):
		self.take_coord = space.chessboard[PChess_move.start()]
		self.drop_coord = space.chessboard[PChess_move.end()]
		self.isEndMove = True
		self.orientation = [2.36, 1.57, -3.14]
		self.take_pose = self.take_coord + self.orientation
		self.drop_pose = self.drop_coord + self.orientation
		if PChess_move.moving_piece().type() != '.':
			self.piece_height = height.pieces_height[PChess_move.moving_piece().type()]
		
	def __str__(self):
		return "Robotic move from " + str(self.take_coord) + " to " + str(self.drop_coord)
		
	
	
	
def create_robotic_move(move):
#Créer x robotic_move à partir d'un move de PChess (chaque robotique_move contient le coup, la hauteur, le endmove)
	
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
	
	
class TestRoboticMove(RoboticMove):
	def __init__(self, PChess_move, h):
		super().__init__(PChess_move)
		self.piece_height = h
	
