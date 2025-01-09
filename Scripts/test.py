from Space import height 
from Robot import Robot
import argparse
import requests
import sys
import PChess as pc
from lichess import get_move

classic_FEN = 'rnbqkbnrpppppppp................................PPPPPPPPRNBQKBNR'
capture_FEN = 'rnbqkbnrppp.pppp...........p........P...........PPPP.PPPRNBQKBNR'
roque_FEN = 'r...k..rpppq.ppp..npbn....b.p.....B.P.....NPBN..PPPQ.PPPR...K..R'
prise_en_passant_FEN = 'rnbqkbnrpppppppp....................P...........PPPP.PPPRNBQKBNR'
promotion_FEN = 'r.b.kbnrpPpp.ppp..n.................p.q..P...N....PPPPPPRNBQKB.R'
fen =  "r.bq.rk.pppp.p....n..p..b....N.....PP.....N.....PPP..PPPR..QKB.R"

parser = argparse.ArgumentParser()
parser.add_argument("--move-to-square", type=str)
parser.add_argument("--obs-pose", action="store_true")
parser.add_argument("--no-flask", action="store_true")
parser.add_argument("--cautious", action="store_true")
parser.add_argument("--no-robot", action="store_true")
parser.add_argument("--lichess",  action="store_true")
args = parser.parse_args()

b = pc.Board(classic_FEN)
b.print()
flask = True

try:
	requests.post("http://127.0.0.1:5000")
except Exception as e:
	print("Error : Flask is not running")
	sys.exit(1)

def send_board_FEN(board):
	if(not flask){return}
	url = "http://127.0.0.1:5000/set-board-FEN"
	payload = {"board_FEN": board.FEN()}
	response = requests.post(url, json=payload)
	if response.status_code == 200:
		print("Board envoyé")
	else:
		print(f"Erreur lors de l'envoi du board : {response.status_code}, {response.text}")

def robot_play(moveStr, cautious = False):
	if len(moveStr) != 4:
		raise Exception("Unvalid Move length")
	
	m = b.create_move(moveStr)
	robot.play_move(b, m, cautious)
	b.play(moveStr)
	
def robot_play_test(moveStr, h):
	if len(moveStr) != 4:
		raise Exception("Unvalid Move length")
	
	m = b.create_move(moveStr)
	robot.play_test_move(m, h)

def send_color_FEN(board):
	if(not flask){return}
	url = "http://127.0.0.1:5000/set-color-FEN"
	payload = {"threats": board.threats(True), 
			"playable": board.playable(True), 
			"controlled": board.controlled(True)}
	
	response = requests.post(url, json=payload)
	if response.status_code == 200:
		print("Color FEN envoyées")
	else:
	    print(f"Erreur lors de l'envoi du board : {response.status_code}, {response.text}")

if args.no_flask:
	flask = False

if args.no_robot:
	send_color_FEN(b)
	send_board_FEN(b)
	while True:
		moveStr = input("Move :")

		if b.play(moveStr) :
			pass
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

		
		if isRobotTurn:
			if args.lichess:
				moveStr = get_move(b.FEN())
			else:
				moveStr = input("Move :")

			robot_play(moveStr, cautious = args.cautious)
		else:
			moveStr = input("Move :")
			b.play(moveStr)
		
		send_color_FEN(b)
		send_board_FEN(b)

		isRobotTurn = not isRobotTurn

	robot.close()

