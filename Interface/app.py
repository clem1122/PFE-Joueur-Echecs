from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)

CORS(app)
# Exemple de données à renvoyer

@app.route('/new-game')
def new_game():
    global color_FEN
    global board_FEN
    color_FEN = {
    "threats":    "................................................................",
    "playable":   "................................................................",
    "controlled": "................................................................",
    "toggle4": "Aide"
}

    board_FEN = "rnbqkbnrpppppppp................................PPPPPPPPRNBQKBNR"

    return "FEN réinitialisées"

@app.route('/')
def home():
    print("La route / a été appelée")
    return "Le serveur flask tourne bien à cette URL"


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


@app.route('/set-color-FEN', methods=['POST'])
def set_color_FEN():
    global color_FEN
    data = request.get_json()
    color_FEN["controlled"] = data["controlled"]
    color_FEN["playable"] = data["playable"]
    color_FEN["threats"] = data["threats"]
    return jsonify({"status": "success", "received": color_FEN})

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
            return jsonify({"error": "No board_FEN available"}), 404
        
        print("Board reçu avec succès")
        return jsonify({"status": "success", "board_FEN": board_FEN}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True)
    new_game()
