import PChess as pc
from Space import space, height
import re

		
class RoboticMove:
	def __init__(self,start_coord,end_coord,piece, isEndMove = True):
		self.take_coord = space.chessboard[start_coord]
		self.drop_coord = space.chessboard[end_coord]
		self.isEndMove = isEndMove
		self.orientation = [2.36, 1.57, -3.14]
		self.take_pose = self.take_coord + self.orientation
		self.drop_pose = self.drop_coord + self.orientation
		if piece.type() != '.':
			self.piece_height = height.pieces_height[piece.type()]
		
	def __str__(self):
		return "Robotic move from " + str(self.take_coord) + " to " + str(self.drop_coord)
		
	
	
def create_robotic_move(move):
#Créer x robotic_move à partir d'un move de PChess (chaque robotique_move contient le coup, la hauteur, le endmove)
	
	return RoboticMove(move.start(), move.end())
		

def create_complex_robotic_move(board,PChess_move):
	A = PChess_move.start()
	B = PChess_move.end()
	played_piece = board.piece_on_square(A)
	
	if(PChess_move.isCapture() and PChess_move.isPromoting()):
		raise Exception("Promotin + Capture")
		
	if(PChess_move.isCapture()):
		killed_piece = board.piece_on_square(B)
		V = valhalla_free_space(board,killed_piece)
		return [RoboticMove(B, V, killed_piece, False), RoboticMove(A, B, played_piece)]
		
	elif(PChess_move.isCastling()):
		move_gap = ord(PChess_move.start()[0]) - ord(PChess_move.start()[0])
		if  move_gap== -2:
			rook_coord = "H" + PChess_move.start()[0]
			new_rook_coord = "F" + PChess_move.start()[0]
		elif move_gap == 2:
			rook_coord = "A" + PChess_move.start()[0]
			new_rook_coord = "D" + PChess_move.start()[0]
		else : 
			raise Exception("Gap between king start and king and equals " + str(move_gap))
			
		return [RoboticMove(A, B), RoboticMove(rook_coord, new_rook_coord)]
		
	elif(PChess_move.isPromoting()):
		F = valhalla(A)
		V = valhalla(A)
		return [RoboticMove(A, V), RoboticMove(F, B)]
		
	elif(PChess_move.isEnPassant()):
		killed_piece_coord = B
		if played_piece.isWhite():
			killed_piece_coord[1] -= 1
		else :
			killed_piece_coord[1] += 1
		
		killed_piece = board.piece_on_square(killed_piece_coord)
			
		V = valhalla_free_space(board,killed_piece)

		return [RoboticMove(A, B), RoboticMove(killed_piece_coord, V)]
		
	else:
		raise Exception("Move is not complex")

def valhalla_free_space(board,killed_piece):

	colour = 'V' if killed_piece.isWhite() else 'v'
	string = board.valhalla_FEN()[0:16] if killed_piece.isWhite() else board.valhalla_FEN()[16:]
	index = string.index('.') +1

	return colour+str(index)
	
def get_castling_rook_coord(board,PChess_move):
	if abs(ord(PChess_move.start()[0]) - ord(PChess_move.start()[0])) == 3:
		return "H" + PChess_move.start()[0]
	if abs(ord(PChess_move.start()[0]) - ord(PChess_move.start()[0])) == 4:
		return "A" + PChess_move.start()[0]



class TestRoboticMove(RoboticMove):
	def __init__(self, PChess_move, h):
		super().__init__(PChess_move)
		self.piece_height = h
	
