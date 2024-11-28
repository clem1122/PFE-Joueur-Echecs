import chess
from datasets import load_dataset

# Function to get the index of the played move
def get_policy_target(board, move_uci):
    # List all legal moves
    legal_moves = list(board.legal_moves)
    
    # Convert the move from UCI to a chess move
    move = chess.Move.from_uci(move_uci)
    
    # Find the index of the played move in the legal moves list
    move_index = legal_moves.index(move)
    
    return move_index

# Example usage
board = chess.Board()
moves_uci = ['e2e4', 'e7e5', 'g1f3', 'b8c6']  # Example UCI moves

for move_uci in moves_uci:
    legal_moves = list(board.legal_moves)
    print(f"Legal moves at this position: {legal_moves}")
    
    move = chess.Move.from_uci(move_uci)
    policy_target = legal_moves.index(move)
    print(f"Policy target for move {move_uci}: {policy_target}")
    
    board.push(move)


# Y a un souciiiiiiiiiiiiis --> LC0 mapping pour legal moves ?
    

dataset = load_dataset('angeluriot/chess_games')

for game in dataset['train']:
    winner = game['winner']
    end_type = game['end_type']
    print(winner)
    print(end_type)
 

def get_value_target(winner):
