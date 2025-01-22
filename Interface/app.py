from flask import Flask, jsonify, request
from flask_cors import CORS
import time
import threading

app = Flask(__name__)

CORS(app)

global have_played_status
have_played_status = False
# Routes de base pour gérer le jeu et la flask

@app.route('/new-game')
def new_game():
    global color_FEN
    global board_FEN
    color_FEN = {
    "threats":    "................................................................",
    "playable":   "................................................................",
    "controlled": "................................................................",
    "protected": "................................................................",
    "help":       "................................................................"
}

    board_FEN = "rnbqkbnrpppppppp................................PPPPPPPPRNBQKBNR"

    return "FEN réinitialisées"

@app.route('/')
def home():
    print("La route / a été appelée")
    return "Le serveur flask tourne bien à cette URL"


# Route pour les boutons à cocher du html

@app.route('/get-info/<toggle_id>', methods=['GET'])
def get_info(toggle_id):
    print(toggle_id)
    # Retourne les infos pour un bouton spécifique
    if toggle_id in color_FEN:
        print(f"Returning info for {toggle_id}: {color_FEN[toggle_id]}")
        result = jsonify({"FEN": color_FEN[toggle_id]})  # Debug
        print("JSON : ", result)
        return result
    else:
        return jsonify({"error": "Bouton non trouvé"}), 404

# Routes pour envoyer et recoir les FEN de couleur
@app.route('/set-color-FEN', methods=['POST'])
def set_color_FEN():
    global color_FEN
    data = request.get_json()
    color_FEN["controlled"] = data["controlled"]
    color_FEN["playable"] = data["playable"]
    color_FEN["threats"] = data["threats"]
    color_FEN["protected"] = data["protected"]
    color_FEN["help"] = data["help"]
    return jsonify({"status": "success", "received": color_FEN})

@app.route('/get-color-FEN', methods=['GET'])
def get_color_FEN():
    print("color FEN envoyée avec succès")
    return jsonify(color_FEN)

#Routes pour envoyer et recevoir le board
@app.route('/set-board-FEN', methods=['POST'])
def set_board_fen():
    global board_FEN 
    try:
        data = request.get_json()
        if "board_FEN" not in data:
            return jsonify({"error": "Key 'board_FEN' is missing"}), 400
        
        print("Payload reçu par flask :", data["board_FEN"])
        board_FEN = data["board_FEN"]
        return jsonify({"status": "success", "received": board_FEN}), 200 
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get-board-FEN', methods=['GET'])
def get_board_fen():
    global board_FEN
    try:
        if 'board_FEN' not in globals() or board_FEN is None:
            return jsonify({"error": "No board_FEN available"}), 501
        
        print("Board reçu avec succès")
        return jsonify({"status": "success", "board_FEN": board_FEN}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

# Routes pour savoir quand est-ce que le joueur humain a joué

@app.route('/set-have-played', methods=['POST'])
def set_have_played():
    global have_played_status
    try:
        data = request.get_json()  # Récupérer les données JSON
        have_played = data.get('have_played')  # Extraire la valeur

        if isinstance(have_played, bool):  # Vérifier que c'est bien un booléen
            have_played_status = have_played
            threading.Thread(target=reset_have_played).start()
            return jsonify({"message": "Statut de have_played mis à jour"}), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/get-have-played', methods=['GET'])
def get_have_played():
    return jsonify(have_played_status), 200
 
 
def reset_have_played(delay=1):
    global have_played_status
    """Réinitialise la variable have_played après un délai."""
    time.sleep(delay)
    have_played_status = False
    print("Statut réinitialisé à False.")


if __name__ == '__main__':
    new_game()
    app.run(debug=True)
    
