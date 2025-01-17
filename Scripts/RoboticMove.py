import Scripts.PChess as pc
from Scripts.Space import space, height
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
		self.piece_height = height.pieces_height[piece.type()] if piece.type() != '.' else 0
		
	def __str__(self):
		return "Robotic move from " + str(self.start_coord) + " to " + str(self.end_coord) + " (Piece : " + str(self.moved_piece) +" ; Hauteur : " + str(self.piece_height) + " )"
		
	
	
def create_robotic_move(move):
#Créer x robotic_move à partir d'un move de PChess (chaque robotique_move contient le coup, la hauteur, le endmove)
	
	return RoboticMove(move.start(), move.end())
		

def create_complex_robotic_move(board, PChess_move, promotion = None):
	A = PChess_move.start()
	B = PChess_move.end()
	played_piece = board.piece_on_square(A)
	
	if(PChess_move.isCapture() and PChess_move.isPromoting()):

		if promotion == None : raise Exception("It is a promotion but no piece type was given")

		killed_piece = board.piece_on_square(B)
		promoted_piece = board.piece_on_square(A)
		V_opponent = valhalla_free_space(board,killed_piece)
		V_self = valhalla_free_space(board,promoted_piece)
		F = get_valhalla_coord(promotion, board)
		resurrected_piece = board.piece_on_square(F)

		return [RoboticMove(B, V_opponent, killed_piece, False), RoboticMove(A, B, played_piece, False), RoboticMove(B, V_self, played_piece, False), RoboticMove(F, B, resurrected_piece)]
		
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

		if promotion == None : raise Exception("It is a promotion but no piece type was given")
		
		V = valhalla_free_space(board,PChess_move.moving_piece())
		F = get_valhalla_coord(promotion,board)

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
	
def choose_promoted_piece(board,isPlayerWhite):
	type_list = get_valhalla_types(board.valhalla_FEN(),isPlayerWhite)
	new_type = None
	while new_type not in type_list : new_type = input("Choisissez votre nouvelle pièce parmi : " + " ".join(type_list) + "\n")

	return new_type

def valhalla_free_space(board,killed_piece):
	colour = 'V' if killed_piece.isWhite() else 'v'
	string = board.valhalla_FEN()[0:19] if killed_piece.isWhite() else board.valhalla_FEN()[19:]
	index_base_10 = string.index('.') +1
	index_base_20 = board.to_base(index_base_10, 20)

	return colour+str(index_base_20)
	
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
	
