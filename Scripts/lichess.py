import re
import chess
import chess.engine

def separate(chaine, position):
    return chaine[:position] + '/' + chaine[position:]
    

def generate_complete_fen(simplified_fen, player_and_castling  = "wKQkq", en_passant = '-'):
    # Initialisation des informations additionnelles
    halfmove_clock = "0"  # Pas de demi-coups joués
    fullmove_number = "1"  # Premier coup
    separated_fen = '/'.join(simplified_fen[i:i+8] for i in range(0, len(simplified_fen), 8))
    slashed_fen = ''
    for i in range(len(separated_fen)-8, -1, -9):
        slashed_fen = re.sub(r"\.+", lambda match: str(len(match.group(0))), separated_fen[i:i+8]) + '/' + slashed_fen

    slashed_fen = slashed_fen[:-1]

    # Construction de la FEN complète
    player = player_and_castling[0]
    castling = '-' if player_and_castling[1:].count('-') == 4 else player_and_castling[1:].replace('-', '')
    complete_fen = f"{slashed_fen} {player} {castling} {en_passant} {halfmove_clock} {fullmove_number}"
    if complete_fen.count('/') != 7:
        raise Exception("Not 7 / : ", complete_fen)
    return complete_fen

# Exemple d'utilisation
simplified_fen = "rnbqkbnrpppppppp................................PPPPPPPPRNBQKBNR"
player_and_castling = "wKQkq"
fen = generate_complete_fen(simplified_fen, player_and_castling)
#print(fen)
#params = {"fen": fen} #"r1bqkbnr/pppppppp/n7/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

#response = requests.get(url, headers=HEADERS, params=params)

#if response.status_code == 200:
#    print(response.json()['pvs'][0]['moves'][:4])
#else:
#    print(f"Erreur {response.status_code}: {response.text}")

def get_stockfish_move(simplified_FEN, player_and_castling = "wKQkq", en_passant = '-', display = True):
    if display : print(simplified_FEN)
    fen = generate_complete_fen(simplified_FEN, player_and_castling, en_passant)
    if display : print("FEN : ",  fen)
    board = chess.Board(fen)
    with chess.engine.SimpleEngine.popen_uci("/mnt/c/Program Files/stockfish/stockfish-windows-x86-64-avx2.exe") as engine:
        # "/mnt/c/Program Files/stockfish/stockfish-windows-x86-64-avx2.exe" Louis
        # "/mnt/d/Programmes/stockfish/stockfish-windows-x86-64-avx2.exe" Clément
    # Request an evaluation of the current position
        result = engine.play(board, chess.engine.Limit(time=1.0))  # Limit the analysis to 1 seconds
        if display : print("coup de stockfish : ", result.move)
    return result.move.uci()

def simplified_FEN(FEN):
    sFEN = ''
    for char in FEN:
        if char in '12345678':
            for i in range(int(char)):
                sFEN += '.'
        elif char == '/':
            continue
        else:
            sFEN += char
    if len(sFEN) != 64:
        raise Exception("FEN size not 64")
    return sFEN

# print(simplified_FEN('R7/3k4/R3pP2/B4P2/8/8/1b6/K7'))
# print(simplified_FEN('r1bqkbnr/pppppppp/n7/8/8/8/PPPPPPPP/RNBQKBNR'))
