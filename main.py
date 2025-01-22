from Scripts import PChess as pc
from Scripts.Space import height 
from Scripts.Robot import Robot
from Scripts.RoboticMove import get_valhalla_coord
from Scripts.lichess import get_stockfish_move

from Vision import calibration
from Vision.calibration import take_picture
from Vision.delete_images import del_im
from Vision.oracle_function import oracle

from sys import exit
import argparse
import requests
import cv2
import signal

pieces_list = ['p','P','n','N','b','B	','r','R','q','Q','k','K']
classic_FEN = 'rnbqkbnrpppppppp................................PPPPPPPPRNBQKBNR'
capture_FEN = 'rnbqkbnrppp.pppp........ ...p........P...........PPPP.PPPRNBQKBNR'
roque_FEN = 'r...k..rpppq.ppp..npbn....b.p.....B.P.....NPBN..PPPQ.PPPR...K..R'
prise_en_passant_FEN = '............p........p.......P...................q......K......k'
promotion_FEN = 'r.b.kbnrpPpp.ppp..n.................p.q..P...N....PPPPPPRNBQKB.R'
promotion_FEN2 = '............P........................p...........K.............k'
fen = 'r.bqkbnr..p..pppp..p....Pp.Pp.......P........N..P.P..PPPRNBQKB.R'

parser = argparse.ArgumentParser()
parser.add_argument("--move-to-square", "-m", type=str)
parser.add_argument("--obs-pose", "-o", action="store_true")
parser.add_argument("--no-flask", "--nf", action="store_true")
parser.add_argument("--cautious", "-c", action="store_true")
parser.add_argument("--no-robot", "--nr", action="store_true")
parser.add_argument("--stockfish", "-s", action="store_true")
parser.add_argument("--take-picture", "--tp", "--tp", nargs="?", const=True)
parser.add_argument("--calibration", action="store_true")
args = parser.parse_args()
isWhite = False
vision = not args.no_robot


g = pc.Game(classic_FEN)
b = g.board
b.print()
flask = not (args.no_flask or args.take_picture)
try:
	imVide = cv2.imread("Images/calibration_img.png")
except e:
	print("Warning : no calibration file")
	

	
if flask:
	try:
		requests.post("http://127.0.0.1:5000")
	except Exception as e:
		print("Error : Flask is not running")
		exit(1)

def have_human_played():
	response = requests.get('http://127.0.0.1:5000/get-have-played')
	response.raise_for_status()
	data = response.json()
	have_played =  data["have_played"]

	return have_played

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
	print("valhalla coord : " + valhalla_coord)
	b.modify_piece(valhalla_coord, '.')

def robot_play(moveStr, cautious = False):
	promotion = None
	if len(moveStr) != 4 and len(moveStr) !=5:
		raise Exception(moveStr + " has an unvalid Move length")
	
	if len(moveStr) == 5:
		if moveStr[4] in pieces_list :
			promotion = moveStr[4] if moveStr[3] == '1' else moveStr[4].upper()
		else : raise Exception(moveStr + " is not a valid 5-length move")
	
	
	m = b.create_move(moveStr[:4])
	if not b.is_legal(m): return False
	robot.play_move(b, m, cautious, promotion)
	g.play(moveStr)
	if m.isPromoting() : manage_promotion(promotion, m)
	return True
	
def robot_play_test(moveStr, h):
	if len(moveStr) != 4:
		raise Exception("Unvalid Move length")
	
	m = b.create_move(moveStr)
	robot.play_test_move(m, h)

def send_color_FEN(board):
	if(not flask):
		return
	spec_rules = "b" +  board.special_rules()[1:]
	best_move = get_stockfish_move(board.FEN(), spec_rules, board.en_passant_coord(), display = False)
	index_1 = board.coord_to_index(best_move[:2])
	index_2 = board.coord_to_index(best_move[2:])
	best_FEN = ['.']*64
	if args.stockfish:
		best_move = get_stockfish_move(board.FEN(), spec_rules, board.en_passant_coord())
		index_1 = board.coord_to_index(best_move[:2])
		index_2 = board.coord_to_index(best_move[2:])
		best_FEN[index_1] = '1'
		best_FEN[index_2] = '1'

	url = "http://127.0.0.1:5000/set-color-FEN"
	payload = {"threats": board.threats(isWhite), 
			"playable": board.playable(isWhite), 
			"controlled": board.controlled(isWhite),
			"protected": board.protected(isWhite),
			"help": best_FEN}
	
	response = requests.post(url, json=payload)
	if response.status_code == 200:
		print("Color FEN envoyées")
	else:
	    print(f"Erreur lors de l'envoi du board : {response.status_code}, {response.text}")

def get_move():
	if args.stockfish:
		return get_stockfish_move(b.FEN(), b.special_rules(), b.en_passant_coord())
	else:
		return input("Move :")

def play(moveStr):
	global isRobotTurn

	if args.no_robot or not isRobotTurn:
		if g.play(moveStr):
			isRobotTurn = not isRobotTurn
			return True
		return False
	else:
		if robot_play(moveStr, cautious = args.cautious):
			playCount = g.play_count()
			isRobotTurn = not isRobotTurn
			return True
		return False

def see(photoId, human = False):
	if args.no_robot: return
	global im_pre_robot, im_post_robot
	if not human:
		im_post_robot = take_picture(robot, photoId)
		origin, end, type, color = oracle(im_pre_robot, im_post_robot, imVide, debug=False)
	else:
		im_pre_robot = take_picture(robot, photoId)
		origin, end, type, color = oracle(im_post_robot, im_pre_robot, imVide, debug=False)

	return origin + end, type, color





if args.calibration:
	calibration.main()
	exit(0)

if args.take_picture:
	if isinstance(args.take_picture, str):
		name = args.take_picture
	else:
		name = 'calibration_img'
	robot = Robot()
	take_picture(robot, name)
	exit(0)


def close(signal_received, frame):
	print("\nSignal d'interruption reçu (Ctrl+C). Fermeture en cours...")
	if not args.no_robot: 
		robot.nyrio.close_connection()

	exit(0)

signal.signal(signal.SIGINT, close)

if args.move_to_square :
	robot = Robot()
	robot.move_to_square(args.move_to_square)
	exit(0)

if args.obs_pose:
	robot = Robot()
	robot.move_to_obs_pose()
	exit(0)

del_im('Images/')

if not args.no_robot: 
	robot = Robot()
	robot.move_to_obs_pose()
	if vision:
		im_pre_robot = take_picture(robot, 0)
		im_post_robot = im_pre_robot


send_board_FEN(b)
send_color_FEN(b)
isRobotTurn = True

while True:	
	playCount = g.play_count() + 1

	if isRobotTurn:
		moveStr = get_move()
		if not play(moveStr): continue

		if vision:
			allegedMove, type, color = see(playCount)
			if allegedMove != moveStr:
				print("Warning : Coup détécté " + allegedMove + " != coup joué " + moveStr)

		# Verification du coup joué par le robot
	else:
		#moveStr = get_move()

		if vision :
			while not have_human_played() :
				pass

		if vision:
			if args.no_flask: input("Entrée quand le coup est joué...")
			allegedMove, type, color = see(playCount, human=True)
			if not play(allegedMove):
				print("Warning : Coup détécté " + allegedMove + " semble être erroné")
				moveStr = get_move()
				while not play(moveStr):
					moveStr = get_move()
		else: 
			#moveStr = get_move()
			moveStr = input("Move : ")
			play(moveStr)
				
		
	send_color_FEN(b)
	send_board_FEN(b)

