from Space import height 
from Robot import Robot
import sys
import argparse
import PChess as pc

parser = argparse.ArgumentParser()
parser.add_argument("--move-to-square", type=str)
args = parser.parse_args()

b = pc.Board()
b.print()
robot = Robot()

def robot_play(moveStr):
	if len(moveStr) != 4:
		raise Exception("Unvalid Move length")
	
	m = b.create_move(moveStr)
	robot.play_move(m)
	b.play(moveStr)
	
def robot_play_test(moveStr, h):
	if len(moveStr) != 4:
		raise Exception("Unvalid Move length")
	
	m = b.create_move(moveStr)
	robot.play_test_move(m, h)
	
	
if args.move_to_square :
	robot.move_to_square(args.move_to_square)
	
else :

	isRobotTurn = True

	while True:
		moveStr = input("Move :")
		if isRobotTurn:
			robot_play(moveStr)
		else:
			b.play(moveStr)
		isRobotTurn = not isRobotTurn

robot.close()

	
