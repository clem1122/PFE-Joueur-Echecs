from Space import height 
from Robot import Robot
from RoboticMove import get_valhalla_coord
import sys
import argparse
import requests
import sys
import PChess as pc
from lichess import get_move

pieces_list = ['p','P','n','N','b','B','r','R','q','Q','k','K']
classic_FEN = 'rnbqkbnrpppppppp................................PPPPPPPPRNBQKBNR'
capture_FEN = 'rnbqkbnrppp.pppp...........p........P...........PPPP.PPPRNBQKBNR'
roque_FEN = 'r...k..rpppq.ppp..npbn....b.p.....B.P.....NPBN..PPPQ.PPPR...K..R'
prise_en_passant_FEN = '............p........p.......P...................q......K......k'
promotion_FEN = 'r.b.kbnrpPpp.ppp..n.................p.q..P...N....PPPPPPRNBQKB.R'
promotion_FEN2 = '............P........................p...........K.............k'
fen = 'r.....k.pQpb..p...nbpq.p.....r.....P.p..P..B.N...PPN.PPPR.B..RK.'

parser = argparse.ArgumentParser()
parser.add_argument("--move-to-square", "-m", type=str)
parser.add_argument("--obs-pose", "-o", action="store_true")
parser.add_argument("--no-flask", "--nf", action="store_true")
parser.add_argument("--cautious", "-c", action="store_true")
parser.add_argument("--no-robot", "--nr", action="store_true")
parser.add_argument("--stockfish", "-s", action="store_true")
args = parser.parse_args()
isWhite = False

g = pc.Game(promotion_FEN2)
b = g.board()
b.print()
flask = not args.no_flask
	
if flask:
	try:
		requests.post("http://127.0.0.1:5000")
	except Exception as e:
		print("Error : Flask is not running")
		sys.exit(1)

def send_board_FEN(board):
	if(not flask):
		return
	
	url = "http://127.0.0.1:5000/set-board-FEN"
	payload = {"board_FEN": board.FEN()}
	response = requests.post(url, json=payload)
	if response.status_code == 200:
		print("Board envoyé")
	else:
		print(f"Erreur lors de l'envoi du board : {response.status_code}, {response.text}")


def manage_promotion(promotion_piece, move):
	if promotion_piece == None : raise Exception("No promotion type")

	print("Demande de promotion sur la case " + move.end() + " en " + promotion_piece)
	b.modify_piece(move.end(), promotion_piece)
	valhalla_coord = get_valhalla_coord(promotion_piece, b)
	b.modify_piece(valhalla_coord, '.')
	# Quand c'est le robot qui joue, l'IA donne le mouvement avec le caractère de la nouvelle pièce
	# Quand c'est le joueur qui joue, la vision donne le coup, PChess voit que c'est une promotion, il photographie le valhalla, connait la pièce partie et voit le nouveau trou


def robot_play(moveStr, cautious = False):
	promotion = None
	if len(moveStr) != 4 and len(moveStr) !=5:
		raise Exception(moveStr + " has an unvalid Move length")
	
	if len(moveStr) == 5:
		if moveStr[4] in pieces_list :
			promotion = moveStr[4] if moveStr[3] == '1' else moveStr[4].upper()
			moveStr = moveStr[:4] + promotion
		else : raise Exception(moveStr + " is not a valid 5-length move")
	
	
	m = b.create_move(moveStr[:4])
	robot.play_move(b, m, cautious, promotion)
	print("move : <" + moveStr + ">")
	result = g.play(moveStr)
	if m.isPromoting() : manage_promotion(promotion, m)
	return result
	
def robot_play_test(moveStr, h):
	if len(moveStr) != 4:
		raise Exception("Unvalid Move length")
	
	m = b.create_move(moveStr)
	robot.play_test_move(m, h)

def send_color_FEN(board):
	if(not flask):
		return
	url = "http://127.0.0.1:5000/set-color-FEN"
	payload = {"threats": board.threats(isWhite), 
			"playable": board.playable(isWhite), 
			"controlled": board.controlled(isWhite)}
	
	response = requests.post(url, json=payload)
	if response.status_code == 200:
		print("Color FEN envoyées")
	else:
	    print(f"Erreur lors de l'envoi du board : {response.status_code}, {response.text}")


if args.no_robot:
	send_color_FEN(b)
	send_board_FEN(b)
	while True:
		moveStr = input("Move :")
		g.print_history()
		b = g.board()
		promotion = None
		if len(moveStr) != 4 and len(moveStr) !=5:
			raise Exception(moveStr + " has an unvalid Move length")
		
		if len(moveStr) == 5:
			if moveStr[4] in pieces_list :
				promotion = moveStr[4]
				moveStr = moveStr[:4]
			else : raise Exception(moveStr + " is not a valid 5-length move")

		m = b.create_move(moveStr)
		if g.play(moveStr):
			if m.isPromoting(): manage_promotion(promotion, m)

			send_color_FEN(b)
			send_board_FEN(b)

elif args.move_to_square :
	robot = Robot()
	robot.move_to_square(args.move_to_square)

elif args.obs_pose:
	robot = Robot()
	robot.move_to_obs_pose()

else:
	robot = Robot()
	send_board_FEN(b)
	isRobotTurn = True

	while True:	
		b = g.board()
		if isRobotTurn:
			if args.stockfish:
				moveStr = get_move(b.FEN(), b.special_rules(), b.en_passant_coord())
			else:
				moveStr = input("Move :")

			if robot_play(moveStr, cautious = args.cautious):
				isRobotTurn = not isRobotTurn
		else:
			moveStr = input("Move :")
			if g.play(moveStr):
				isRobotTurn = not isRobotTurn
		
		send_color_FEN(b)
		send_board_FEN(b)

	robot.close()

