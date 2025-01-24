from Scripts import PChess as pc
from Scripts.Space import height 
from Scripts.Robot import Robot
from Scripts.RoboticMove import get_valhalla_coord
from Scripts.lichess import get_stockfish_move

from Vision import calibration
from Vision.calibration import take_picture
from Vision.delete_images import del_im, del_pkl
from Vision.oracle_function import oracle
from Vision.check_valhalla import check_valhalla

from sys import exit
import argparse
import requests
import cv2
import signal
from time import sleep

pieces_list = ['p','P','n','N','b','B	','r','R','q','Q','k','K']
classic_FEN = 'rnbqkbnrpppppppp................................PPPPPPPPRNBQKBNR'
capture_FEN = 'rnbqkbnrppp.pppp........ ...p........P...........PPPP.PPPRNBQKBNR'
roque_FEN = 'r...k..rpppq.ppp..npbn....b.p.....B.P.....NPBN..PPPQ.PPPR...K..R'
prise_en_passant_FEN = '............p........p.......P...................q......K......k'
promotion_FEN = 'r.b.kbnrpPpp.ppp..n.................p.q..P...N....PPPPPPRNBQKB.R'
promotion_FEN2 = '............P........................p...........K.............k'
promotion_FEN3 = '.nbqkbn..ppppppp................................pPPPPPPP.NBQKBN.'
fen = 'r.bqkbnr..p..pppp..p....Pp.Pp.......P........N..P.P..PPPRNBQKB.R'

parser = argparse.ArgumentParser()
parser.add_argument("--move-to-square", "-m", type=str)
parser.add_argument("--obs-pose", "-o", action="store_true")
parser.add_argument("--V-pose", "--oV", action="store_true")
parser.add_argument("--v-pose", "--ov", action="store_true")
parser.add_argument("--no-flask", "--nf", action="store_true")
parser.add_argument("--cautious", "-c", action="store_true")
parser.add_argument("--no-robot", "--nr", action="store_true")
parser.add_argument("--stockfish", "-s", action="store_true")
parser.add_argument("--take-picture", "--tp", nargs="?", const=True)
parser.add_argument("--calibration", action="store_true")
args = parser.parse_args()
isWhite = False
vision = not args.no_robot

is_human_white = False
g = pc.Game(classic_FEN)
b = g.board
b.print()
flask = not (args.no_flask or args.take_picture)

try:
	imVide = cv2.imread("Images/calibration_img.png")
except:
	print("Warning : no calibration file")

if flask:
	try:
		requests.post("http://127.0.0.1:5000")
	except Exception as e:
		print("Error : Flask is not running")
		exit(1)

def get_human_promotion_move(move, isWhite):

	if move.isPromoting():
		print("Le move " + move.start() + move.end() + " est une promotion")
		emptied_square = valhalla_see(isWhite)
		new_piece = b.piece_on_square(emptied_square)
		correct_move = move.start() + move.end() + new_piece.type().lower()
		print("Le move corrigé est donc " + correct_move)
		return correct_move

	else :
		return move.start() + move.end()



def have_human_played():
	requests.post('http://127.0.0.1:5000/reset-have-played')
	response = requests.get('http://127.0.0.1:5000/get-have-played')
	response.raise_for_status()
	data = response.json()
	have_played =  data["have_played"]

	return have_played

def send_board_FEN(board):
	if(not flask):
		return
	
	url = "http://127.0.0.1:5000/set-board-FEN"
	payload = {"board_FEN": board.FEN(),
				"valhalla": board.valhalla_FEN()}
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
	best_FEN = ['.']*64
	if args.stockfish:
		best_move = get_stockfish_move(board.FEN(), spec_rules, board.en_passant_coord())
		if best_move == None: return
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

def send_state(board):
	
	url = "http://127.0.0.1:5000/set-state"
	whiteKingSquare = board.index_to_coord(board.find_king(True))
	blackKingSquare = board.index_to_coord(board.find_king(False))

	payload = {
		"check": board.is_check(True, whiteKingSquare), 
		"checkmate": board.is_checkmate(True), 
		"checked": board.is_check(False, blackKingSquare),
		"checkmated": board.is_checkmate(False)
	}	
	response = requests.post(url, json=payload)
	if response.status_code == 200:
		print("State envoyé")
	else:
	    print(f"Erreur lors de l'envoi du state : {response.status_code}, {response.text}")
	

def get_move():
	if args.stockfish:
		return get_stockfish_move(b.FEN(), b.special_rules(), b.en_passant_coord())
	else:
		return input("Move :")

def reask_for_move(first_move) :
	answer = input("Le coup détécté (" + first_move + ") n'est pas légal. Est-ce ce que vous vouliez jouer ? [o/n]")
	while answer != "o" and answer != "n" :
		answer = input("Le coup détécté (" + first_move + ") n'est pas légal. Est-ce ce que vous vouliez jouer ? [o/n]")
	
	if answer == "o" : 
		new_move = input("Ce coup n'est pas légal parce que [insérez explication]. Remets ta pièce où elle était et écris un coup légal. ")
	if answer == "n" :
		new_move = input("Quel est le coup que tu as joué sur l'échiquier alors ? ")

	return new_move

def play(moveStr):
	global isRobotTurn
	print("Le joueur essaie de jouer " + moveStr)
	if args.no_robot or not isRobotTurn:
		move = b.create_move(moveStr)
		moveStr = get_human_promotion_move(move, is_human_white)
		print("On essaie de jouer le move corrigé " + moveStr)
		if g.play(moveStr):
			if move.isPromoting() : 
				promo_type = moveStr[4].upper() if move.moving_piece().isWhite() else moveStr[4]
				print("Promotion pour prendre " + promo_type)
				manage_promotion(promo_type, move)
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
		sleep(0.5)
		im_post_robot = take_picture(robot, photoId)
		origin, end = oracle(im_pre_robot, im_post_robot, imVide, debug=True)
	else:
		sleep(0.5)
		im_pre_robot = take_picture(robot, photoId)
		origin, end = oracle(im_post_robot, im_pre_robot, imVide, debug=True)

	return origin + end

def valhalla_see(isWhite):
	if isWhite : 
		string = "V"
		robot.move_to_V_pose()
	else : 
		string = "v"
		robot.move_to_v_pose()

	try:
		reference_valhalla = cv2.imread("Images/" + string + "_calibration_img.png")
	except e:
		print("Warning : no valhalla calibration file named " + "Images/" + string + "_calibration_img.png")

	new_valhalla = take_picture(robot, "valhalla")
	v_index = int(check_valhalla(new_valhalla,reference_valhalla,isWhite))
	v_index_good_base = b.to_base(v_index,20)

	return string + v_index_good_base



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
		robot.niryo.close_connection()
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

if args.V_pose:
	robot = Robot()
	robot.move_to_V_pose()
	exit(0)

if args.v_pose:
	print("v_pose")
	robot = Robot()
	robot.move_to_v_pose()
	exit(0)

del_im('Images/')

if not args.no_robot: 
	robot = Robot()
	robot.move_to_obs_pose()
	if vision:
		sleep(0.5)
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
			allegedMove = see(playCount)
			if allegedMove != moveStr:
				print("Warning : Coup détécté " + allegedMove + " != coup joué " + moveStr)

		# Verification du coup joué par le robot
	else:
		#moveStr = get_move()

		if vision:
			if args.no_flask: input("Entrée quand le coup est joué...")
			else : have_human_played()
			allegedMove = see(playCount, human=True)

			while not play(allegedMove):
				print(allegedMove)
				allegedMove = reask_for_move(allegedMove)
				print(allegedMove)
		else: 
			#moveStr = get_move()
			moveStr = input("Move : ")
			play(moveStr)
				
		
	send_color_FEN(b)
	send_board_FEN(b)
	send_state(b)

