from Scripts import PChess as pc
from Scripts.Space import height 
from Scripts.Robot import Robot
from Scripts.RoboticMove import get_valhalla_coord, RoboticMove
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

def ecrire_lignes(nom_fichier, ligne1, ligne2):
    with open(nom_fichier, 'w') as f:
        f.write(ligne1 + '\n')
        f.write(ligne2 + '\n')

def lire_lignes(nom_fichier):
    with open(nom_fichier, 'r') as f:
        lignes = f.readlines()
    return [ligne.strip() for ligne in lignes]



pieces_list = ['p','P','n','N','b','B	','r','R','q','Q','k','K']
piece_dictionnary = {"p" : "pion", "P" : "pion", "n" : "cavalier", "N" : "cavalier", "b" : "fou",  "B" : "fou", "r" : "tour", "R" : "tour", "k" : "roi", "K" : "roi", "q" : "dame", "Q" : "dame"}
classic_FEN = 'rnbqkbnrpppppppp................................PPPPPPPPRNBQKBNR'
h8_FEN = 'RnbqkbnR.pppppp.................................PPPPPPPPRNBQKBNR'
capture_FEN = 'rnbqkbnrppp.pppp........ ...p........P...........PPPP.PPPRNBQKBNR'
roque_FEN = 'r...k..rpppq.ppp..npbn....b.p.....B.P.....NPBN..PPPQ.PPPR...K..R'
prise_en_passant_FEN = '............p........p.......P...................q......K......k'
promotion_FEN = 'r.b.kbnrpPpp.ppp..n.................p.q..P...N....PPPPPPRNBQKB.R'
promotion_FEN2 = '............P........................p...........K.............k'
promotion_FEN3 = '.nbqkbn..ppppppp................................pPPPPPPP.NBQKBN.'
fen = 'r.bqkbnr..p..pppp..p....Pp.Pp.......P........N..P.P..PPPRNBQKB.R'
classic_valhalla_FEN = 'QRBN...............qrbn...............'

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
parser.add_argument("--didacticiel", "-d", action="store_true")
parser.add_argument("--victory", action="store_true")
parser.add_argument("--reset", action="store_true")
parser.add_argument("--backup", action="store_true")

args = parser.parse_args()
board_FEN = classic_FEN
board_valhalla_FEN = classic_valhalla_FEN

backup_file = "backup.txt"
if args.backup :
	[board_FEN, board_valhalla_FEN] = lire_lignes(backup_file)
isWhite = False
vision = not args.no_robot

is_human_white = False
g = pc.Game(board_FEN, board_valhalla_FEN)
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
		if best_move == None: best_FEN = ['.']*64
		index_1 = board.coord_to_index(best_move[:2])
		index_2 = board.coord_to_index(best_move[2:])
		best_FEN[index_1] = '1'
		best_FEN[index_2] = '1'

	checking_pieces = ['.']*64
	if board.checking(isWhite).count('1') != 1:
		checking_pieces = board.checking(isWhite)

	url = "http://127.0.0.1:5000/set-color-FEN"
	payload = {"threats": board.threats(isWhite), 
			"playable": board.playable(isWhite), 
			"controlled": board.controlled(isWhite),
			"protected": board.protected(isWhite),
			"help": best_FEN,
			"checking": checking_pieces}
	
	response = requests.post(url, json=payload)
	if response.status_code == 200:
		print("Color FEN envoyées")
	else:
	    print(f"Erreur lors de l'envoi du board : {response.status_code}, {response.text}")

def send_state(board, unsure = ""):
	
	url = "http://127.0.0.1:5000/set-state"
	whiteKingSquare = board.index_to_coord(board.find_king(True))
	blackKingSquare = board.index_to_coord(board.find_king(False))

	payload = {
		"check": board.is_check(True, whiteKingSquare), 
		"checkmate": board.is_checkmate(True), 
		"checked": board.is_check(False, blackKingSquare),
		"checkmated": board.is_checkmate(False),
		"unsure" : unsure
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
		origin, end = oracle(im_pre_robot, im_post_robot, imVide, debug=False)
	else:
		sleep(0.5)
		im_pre_robot = take_picture(robot, photoId)
		origin, end = oracle(im_post_robot, im_pre_robot, imVide, debug=False)

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

def didac_move(board, robot, start_square,end_square, end_move = False) : 
	rob_move = RoboticMove(start_square, end_square, board.piece_on_square(start_square),end_move)
	board.modify_piece(end_square,board.piece_on_square(start_square).type())
	board.modify_piece(start_square,'.')
	robot.execute_move(rob_move)

	send_color_FEN(board)
	send_board_FEN(board)
	send_state(board)

def say(robot, text):
	requests.post("http://127.0.0.1:5000/set-message", json={"message":text})
	robot.niryo.say(text, 1)
	print(text)
	
def sequence_didacticiel():
	
	FEN_vide = '................................................................'
	valhalla_FEN = 'QRBNKP.............qrbnkp.............'
	robot = Robot()
	b = pc.Board(FEN_vide,valhalla_FEN)
	b.print()
	send_color_FEN(b)
	send_board_FEN(b)
	send_state(b)

	# BASE
	robot.move_to_obs_pose()
	say(robot, "Coucou ! Je suis Nini, un robot pour t'apprendre à jouer aux échecs ! Apprenons les règles de base. ")
	say(robot, "Mets les pièces dans le cimetière comme montré sur l'écran. Une fois fait, appuies sur le bouton.")
	have_human_played()
	say(robot, "Ceci est le plateau de jeu, composé de 64 cases, moitié blanches moitiées noires.")
	robot.move_to_square("d8")
	say(robot, "Les colonnes sont indiquées par des lettres.")
	robot.move_to_square("h4")
	say(robot, "Les lignes sont indiquées par des chiffres.")
	robot.move_to_square("d4")
	say(robot, "Une case se définit en donnant sa colonne puis sa ligne : cette case est par exemple la case d4.")
	say(robot, "Voyons maintenant les pièces.")
	say(robot, "Lorsque je suis en haut, appuie sur le bouton pour voir la suite. ")
	have_human_played()

	# ROI
	didac_move(b, robot,"V5","b2")
	say(robot, "Voici le roi blanc.")
	didac_move(b, robot,"v5","g7")
	say(robot, "Et voici le roi noir.")
	say(robot, "Les rois ne peuvent se déplacer que d'une case, mais dans toutes les directions.")
	say(robot, "Clique sur la coche Coups Possibles pour voir les cases accessibles.")
	robot.move_to_obs_pose()
	#say(robot, "Appuies sur le bouton pour voir la suite. ")
	have_human_played()

	# TOUR
	say(robot, "Voyons maintenant une autre pièce, la tour.")
	didac_move(b, robot,"v2","d4")
	say(robot, "La tour se déplace d'autant de cases que l'on veut le long d'une ligne ou d'une colonne.")
	didac_move(b, robot,"d4","d1")
	say(robot, "Aux échecs cependant les pièces peuvent être bloquées dans leur mouvement.")
	didac_move(b, robot,"v6","d7")
	say(robot, "Les pièces alliées ne peuvent être traversées.")
	didac_move(b, robot,"V2","f4")
	say(robot, "Les pièces ennemies non plus.") 
	say(robot, "Arriver sur la case d'un adversaire permet de la prendre. Comme ceci.")
	didac_move(b, robot,"d4","v2")
	didac_move(b, robot,"f4","d4", True)

	# FOU
	#say(robot, "Appuies sur le bouton pour voir la suite. ")
	have_human_played()
	didac_move(b, robot,"d4","V2")
	didac_move(b, robot,"d7","v6")
	say(robot, "Voyons maintenant d'autres pièces.")
	didac_move(b, robot,"v3","c4")
	say(robot, "Ceci est un fou.") 
	didac_move(b, robot,"c4","e2")
	say(robot, "Il peut se déplacer d'autant de cases qu'il veut mais uniquement sur les diagonales.")
	didac_move(b, robot,"e2","v3", True)
	have_human_played()
	
	# DAME
	didac_move(b, robot, "v1", "c4")
	say(robot, "Voyons maintenant la dame.")
	say(robot, "C'est comme un mélange entre une tour et un fou. Elle peut se déplacer en diagonale...")
	didac_move(b, robot, "c4", "g1")
	say(robot, "... ou le long d'une ligne ou d'une colonne.")
	didac_move(b, robot, "g1", "g5", True)
	say(robot, "Prends-en soin, c'est ta pièce la plus forte !")
	#say(robot, "Appuies sur le bouton pour voir la suite. ")
	have_human_played()

	# CAVALIER
	didac_move(b, robot,"g5","v1")
	say(robot, "Et maintenant, le cavalier. Son mouvement est un peu particulier puisqu'il bouge en L.")
	didac_move(b, robot,"v4","d4")
	say(robot, "Il avance de deux cases dans une direction, puis de une case à sa droite ou à sa gauche.")
	didac_move(b, robot,"V6","d5")
	say(robot, "Il a aussi une autre particularité : les autres pièces ne gênent pas son déplacement.")
	didac_move(b, robot,"V1","c6")
	say(robot, "Sa case d'arrivée doit cependant être vide, ou contenir un adversaire à prendre.")
	didac_move(b, robot,"c6","V1")
	didac_move(b, robot,"d4","c6", True)
	#say(robot, "Appuies sur le bouton pour voir la suite. ")
	have_human_played()
	didac_move(b, robot,"c6","v4")
	didac_move(b, robot,"d5","e3")
	

	# PION 

	say(robot, "Terminons par le pion.")
	didac_move(b, robot,"v6","d7")
	say(robot, "Celui-ci peut se placer uniquement vers l'avant.")
	say(robot, "S'il se situe sur sa case de départ, il peut avancer de une ou 2 cases.")
	didac_move(b, robot,"d7","d5")
	say(robot, "Dans le cas contraire, il ne peut se déplacer que de une case à la fois.")
	didac_move(b, robot,"e3","e4")
	say(robot, "Enfin, le pion ne peut capturer une pièce que de une case en diagonale, comme ceci.")
	didac_move(b, robot,"e4","V6")
	didac_move(b, robot,"d5","e4", True)
	#say(robot, "Appuies sur le bouton pour voir la suite. ")
	have_human_played()

	say(robot, "Mais comment faire pour gagner ?")
	say(robot, "Déjà, quand un roi peut être pris par une pièce, on dit qu'il est échec.")
	didac_move(b, robot,"v2","b7")
	say(robot, "Ici, la tour est en capacité de prendre le roi.")
	say(robot, "Un joueur ne peut jamais finir son tour avec son roi en échec.")
	say(robot, "Soit il le bouge pour le mettre hors de danger...")
	didac_move(b, robot,"b2","c2")
	say(robot, "... soit il intercale une pièce pour faire bouclier...")
	didac_move(b, robot,"b7","c7")
	didac_move(b, robot,"V2","c4")
	say(robot, "... soit il élimine la menace.")
	didac_move(b, robot,"c7","v2")
	didac_move(b, robot,"c4","c7", True)
	have_human_played()
	didac_move(b, robot,"c7","V2")
	say(robot, "Lorsqu'un joueur est en échec et qu'il n'a aucun moyen de l'enlever")
	say(robot, "on dit qu'il est échec et mat.")
	didac_move(b, robot,"v1","c3")
	didac_move(b, robot,"v2","g1")
	didac_move(b, robot,"v3","e5")
	say(robot, "Voici un exemple.")
	robot.move_to_obs_pose()
	robot.move_to_square("c2", height.POINT)
	say(robot, "Le roi blanc est ici.")
	robot.move_to_obs_pose()
	robot.move_to_square("c3", height.POINT)
	say(robot, "Il est mis en échec par la reine noire.")
	robot.move_to_obs_pose()
	robot.move_to_square("g1", height.POINT)
	say(robot, "Il ne peut se déplacer nulle part à cause de la reine et de la tour.")
	robot.move_to_obs_pose()
	robot.move_to_square("e5", height.POINT)
	say(robot, "Et il ne peut pas prendre la dame : le fou le mettrait en échec.")
	say(robot, "Le roi est alors dit échec et mat.")

	return 

if args.victory:
	robot = Robot()
	robot.execute_registered_trajectory("dance11")
	robot.niryo.move_to_home_pose()
	exit(0)

if args.didacticiel:
	robot = Robot()
	sequence_didacticiel(robot, b)
	exit(0)

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


def save_backup(FEN,valhalla_FEN):
	ecrire_lignes(backup_file,FEN,valhalla_FEN)

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

while not g.isOver():	
	playCount = g.play_count() + 1
	unsure = ""

	if isRobotTurn:
		moveStr = get_move()
		if not play(moveStr): continue
		if vision:
			allegedMove = see(playCount)
			if allegedMove != moveStr:
				print("Warning : Coup détécté " + allegedMove + " != coup joué " + moveStr)

		print("Save dans backup")
		save_backup(b.FEN(),b.valhalla_FEN())

		# Verification du coup joué par le robot
	else:
		#moveStr = get_move()

		if vision:
			if args.no_flask: input("Entrée quand le coup est joué...")
			else : have_human_played()
			allegedMove = see(playCount, human=True)

			#Try to reverse the move
			if not play(allegedMove):
				reverseMove = allegedMove[2:] + allegedMove[:2]
				if len(allegedMove) == 5 : reverseMove += allegedMove[4]
				print("Coup détecté non légal, on essaie de jouer l'inverse : " + reverseMove)
				if not play(reverseMove):
					print("Mauvaise détection dans les deux sens. Demande au joueur.")
					while not play(allegedMove):
						unsure = allegedMove
						send_state(b, unsure)
						if args.no_flask:
							allegedMove = input("Ecris-moi ton move (qui doit être légal) : ")
						else:
							data = requests.get("http://127.0.0.1:5000/get-answer")
							allegedMove = data.json()['reponse']
							
					take_picture(robot, playCount)


		else: 
			#moveStr = get_move()
			moveStr = input("Move : ")
			play(moveStr)
				
		
	send_color_FEN(b)
	send_board_FEN(b)
	send_state(b)

