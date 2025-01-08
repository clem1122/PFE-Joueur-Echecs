import os
from dotenv import load_dotenv

load_dotenv()

API_TOKEN = os.getenv("LICHESS_API_TOKEN")

# Préparer les en-têtes avec le jeton API
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}

# Exemple d'utilisation avec une requête
import requests
url = "https://lichess.org/api/cloud-eval"


def separate(chaine, position):
    return chaine[:position] + '/' + chaine[position:]
    

def generate_complete_fen(simplified_fen, player_and_castling):
    # Initialisation des informations additionnelles
    en_passant = "-"  # Aucun pion éligible à la prise en passant
    halfmove_clock = "0"  # Pas de demi-coups joués
    fullmove_number = "1"  # Premier coup

    # Conversion de la FEN simplifiée en FEN standard
    standard_fen = ""
    empty_count = 0

    for char in simplified_fen:
        if char == ".":
            empty_count += 1
            if empty_count == 8:
                standard_fen += '8'
                empty_count = 0
        else:
            if empty_count > 0:
                standard_fen += str(empty_count)
                empty_count = 0
            standard_fen += char
    if empty_count > 0:
        standard_fen += str(empty_count)

    count = 0
    separated_fen = standard_fen
    for charIndex in range(len(standard_fen)-1, 0, -1):
        char = standard_fen[charIndex]
        if char in "12345678":
            count += int(char)
        else:
            count += 1
        if count == 8:
            separated_fen = separate(separated_fen, charIndex)
            
            count = 0
    # Extraction du joueur actif et des droits de roque
    player = player_and_castling[0]
    castling = player_and_castling[1:] if len(player_and_castling) > 1 else "-"

    # Construction de la FEN complète
    complete_fen = f"{separated_fen} {player} {castling} {en_passant} {halfmove_clock} {fullmove_number}"
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

def get_move(simplified_FEN, player_and_castling = "wKQkq"):
    fen = generate_complete_fen(simplified_FEN, player_and_castling)
    params = {"fen": fen} #"r1bqkbnr/pppppppp/n7/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    response = requests.get(url, headers=HEADERS, params=params)

    if response.status_code == 200:
        return response.json()['pvs'][0]['moves'][:4]
    else:
        print(f"Erreur {response.status_code}: {response.text}")
        return 0
