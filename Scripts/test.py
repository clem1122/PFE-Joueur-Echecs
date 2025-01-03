from Space import height 
from Robot import Robot
import sys
import argparse
import requests
import PChess as pc

classic_FEN = 'rnbqkbnrpppppppp................................PPPPPPPPRNBQKBNR'
capture_FEN = 'rnbqkbnrppp.pppp...........p........P...........PPPP.PPPRNBQKBNR'
roque_FEN = 'r...k..rpppq.ppp..npbn....b.p.....B.P.....NPBN..PPPQ.PPPR...K..R'
prise_en_passant_FEN = 'rnbqkbnrpppppppp....................P...........PPPP.PPPRNBQKBNR'
promotion_FEN = 'r.b.kbnrpPpp.ppp..n.................p.q..P...N....PPPPPPRNBQKB.R'

parser = argparse.ArgumentParser()
parser.add_argument("--move-to-square", type=str)
parser.add_argument("--obs-pose", action="store_true")
parser.add_argument("--no-flask", action="store_true")
parser.add_argument("--cautious", action="store_true")
args = parser.parse_args()

b = pc.Board(classic_FEN)
b.print()
robot = Robot()
flask = True

if flask :
	requests.get("http://127.0.0.1:5000/new-game") #Regénère les FEN de la flask

def send_board_FEN(board):
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
if args.move_to_square :
	robot.move_to_square(args.move_to_square)
elif args.obs_pose:
	robot.move_to_obs_pose()	
else :

	isRobotTurn = True

	while True:
		moveStr = input("Move :")
		
		if isRobotTurn:
			robot_play(moveStr, cautious = args.cautious)
		else:
			b.play(moveStr)
		
		if flask:	
			send_color_FEN(b)
			send_board_FEN(b)

		isRobotTurn = not isRobotTurn

robot.close()

