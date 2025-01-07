import PChess as pc
from Space import space, height
import re

		
class RoboticMove:
	def __init__(self,start_coord,end_coord,piece, isEndMove = True):
		self.take_coord = space.chessboard[start_coord]
		self.drop_coord = space.chessboard[end_coord]
		self.start_coord = start_coord
		self.end_coord = end_coord
		self.moved_piece = piece.type()
		self.isEndMove = isEndMove
		self.orientation = [2.36, 1.57, -3.14]
		self.take_pose = self.take_coord + self.orientation
		self.drop_pose = self.drop_coord + self.orientation
		if piece.type() != '.':
			self.piece_height = height.pieces_height[piece.type()]
		
	def __str__(self):
		return "Robotic move from " + str(self.start_coord) + " to " + str(self.end_coord) + " (Piece : " + str(self.moved_piece) +" ; Hauteur : " + str(self.piece_height) + " )"
		
	
	
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
		move_gap = ord(PChess_move.start()[0]) - ord(PChess_move.end()[0])
		if  move_gap== -2:
			rook_coord = "h" + PChess_move.start()[1]
			new_rook_coord = "f" + PChess_move.start()[1]
		elif move_gap == 2:
			rook_coord = "a" + PChess_move.start()[1]
			new_rook_coord = "d" + PChess_move.start()[1]
		else : 
			raise Exception("Gap between king start and king end equals " + str(move_gap))
			
		rook = board.piece_on_square(rook_coord)
			
		return [RoboticMove(A, B, played_piece, False), RoboticMove(rook_coord, new_rook_coord, rook)]
		
	elif(PChess_move.isPromoting()):
		
		type_list = get_valhalla_types(board.valhalla_FEN(),PChess_move.moving_piece().isWhite())
		new_type = None
		while new_type not in type_list : new_type = input("Choisissez votre nouvelle pièce parmi : " + " ".join(type_list) + "\n")
		
		V = valhalla_free_space(board,PChess_move.moving_piece())
		F = get_valhalla_coord(new_type,board)

		return [RoboticMove(A, B, played_piece, False), RoboticMove(B, V, played_piece, False), RoboticMove(F, B, board.piece_on_square(F))]
		
	elif(PChess_move.isEnPassant()):
		killed_piece_coord = B[0]
		row_int = int(B[1])
		if played_piece.isWhite():
			killed_piece_coord += str(row_int - 1)
		else :
			killed_piece_coord += str(row_int + 1)
		
		killed_piece = board.piece_on_square(killed_piece_coord)
			
		V = valhalla_free_space(board,killed_piece)

		return [RoboticMove(A, B, played_piece, False), RoboticMove(killed_piece_coord, V, killed_piece)]
		
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

def get_valhalla_types(valhalla_FEN,isWhite):
	liste_type = []
	for piece in valhalla_FEN:
		if piece.isupper() and isWhite and piece != '.':
			liste_type.append(piece)
		elif piece.islower() and not isWhite and piece != '.':
			liste_type.append(piece)
	
	return liste_type

def get_valhalla_coord(piece_type,board):
	for i in range(len(board.valhalla_FEN())) :
		piece = board.valhalla_FEN()[i]
		if piece == piece_type:
			return board.valhalla_index_to_coord(i)

class TestRoboticMove(RoboticMove):
	def __init__(self, PChess_move, h):
		super().__init__(PChess_move)
		self.piece_height = h
	
