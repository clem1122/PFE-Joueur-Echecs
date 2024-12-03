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
print(board)
moves_uci = ['e2e4', 'e7e5', 'g1f3', 'b8c6']  # Example UCI moves

for move_uci in moves_uci:
    legal_moves = list(board.legal_moves)
    print(f"Legal moves at this position: {legal_moves}")
    
    move = chess.Move.from_uci(move_uci)
    policy_target = legal_moves.index(move)
    print(f"Policy target for move {move_uci}: {policy_target}")
    
    board.push(move)
    print(board)
    

# dataset = load_dataset('angeluriot/chess_games')

# for game in dataset['train']:
#     winner = game['winner']
#     print(winner)
 
########################################################

# CA C'EST BON, en partant du principe qu'on dit au robot que la partie est gagnée quand c'est white qui gagne et perdue quand c'est black qui gagne

def get_value_target(winner):
    if winner == "white":
        return [1, 0, 0]
    if winner == "black":
        return [0, 0, 1]
    else :
        return [0, 1, 0]
    


def generate_policy_mapping():
    policy_map = {}
    index = 0
    
    # Iterate over all source squares
    for from_square in range(64):
        for to_square in range(64):
            # Skip illegal moves (like moving a piece off the board)
            if from_square == to_square:
                continue
            
            # Standard moves
            move_uci = f"{chess.square_name(from_square)}{chess.square_name(to_square)}"
            policy_map[move_uci] = index
            index += 1
            
            # Add promotion moves
            if chess.square_rank(from_square) in [1, 6]:  # If it's a pawn's promotion rank
                for promo in "qrbn":
                    promo_move_uci = f"{chess.square_name(from_square)}{chess.square_name(to_square)}{promo}"
                    policy_map[promo_move_uci] = index
                    index += 1

    return policy_map
print("POLICY MAP")
policy_map = generate_policy_mapping()
print(policy_map)