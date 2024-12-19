import os
from dotenv import load_dotenv

load_dotenv()

API_TOKEN = os.getenv("LICHESS_API_TOKEN")

# Préparer les en-têtes avec le jeton API
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}

# Exemple d'utilisation avec une requête
import requests
url = "https://lichess.org/api/cloud-eval"
params = {"fen": "r1bqkbnr/pppppppp/n7/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"}

#response = requests.get(url, headers=HEADERS, params=params)


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
        else:
            if empty_count > 0:
                standard_fen += str(empty_count)
                empty_count = 0
            standard_fen += char
    if empty_count > 0:
        standard_fen += str(empty_count)

    print(standard_fen)
    # Ajout des séparateurs de rangées "/"
    standard_fen = '/'.join(standard_fen[i:i+8] for i in range(0, len(standard_fen), 8))

    # Extraction du joueur actif et des droits de roque
    player = player_and_castling[0]
    castling = player_and_castling[1:] if len(player_and_castling) > 1 else "-"

    # Construction de la FEN complète
    complete_fen = f"{standard_fen} {player} {castling} {en_passant} {halfmove_clock} {fullmove_number}"
    return complete_fen

# Exemple d'utilisation
simplified_fen = "rnbqkbnrpppppppp................................PPPPPPPPRNBQKBNR"
player_and_castling = "wKQkq"
fen = generate_complete_fen(simplified_fen, player_and_castling)
print(fen)

#if response.status_code == 200:
#    print(response.json())
#else:
#    print(f"Erreur {response.status_code}: {response.text}")
