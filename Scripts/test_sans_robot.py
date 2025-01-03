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
parser.add_argument("--no-flask", action="store_true")
args = parser.parse_args()

b = pc.Board()
b.print()

flask = True

if flask :
	requests.get("http://127.0.0.1:5000/new-game") #Regénère les FEN de la flask

def send_board_FEN(board):
    url = "http://127.0.0.1:5000/set-board-FEN"
    payload = {"board_FEN": board.FEN()} 
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        print("Board envoyé avec succès.")
    else:
        print(f"Erreur lors de l'envoi du board : {response.status_code}, {response.text}")

def send_color_FEN(board):

	url = "http://127.0.0.1:5000/set-color-FEN"
	payload = {"threats": board.threats(True), 
			"playable": board.playable(True), 
			"controlled": board.controlled(True)}

	response = requests.post(url, json=payload)

	if response.status_code == 200:
		print("Color FEN envoyées")
		
		
send_color_FEN(b)
send_board_FEN(b)
while True:
	moveStr = input("Move :")

	if b.play(moveStr) :
		send_color_FEN(b)
		send_board_FEN(b)





