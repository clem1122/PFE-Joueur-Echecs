from flask import Flask, jsonify, Response, request, stream_with_context
from flask_cors import CORS
import time

app = Flask(__name__)

CORS(app)

have_played_status = False
last_click_time = 0

# Routes de base pour gérer le jeu et la flask

@app.route('/new-game')
def new_game():
    global color_FEN
    global board_FEN
    global state
    color_FEN = {
        "threats":    "................................................................",
        "playable":   "................................................................",
        "controlled": "................................................................",
        "protected": "................................................................",
        "help":       "................................................................",
        "checking" :  "................................................................"
    }

    board_FEN = "rnbqkbnrpppppppp................................PPPPPPPPRNBQKBNR"
    valhalla_FEN = "QRBN...............qrbn..............."


    state = {
        "check": False,
        "checkmate": False,
        "checked": False,
        "checkmated": False,
        "unsure" : "",
    }
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
    color_FEN["checking"] = data["checking"]
    return jsonify({"status": "success", "received": color_FEN})

@app.route('/get-color-FEN', methods=['GET'])
def get_color_FEN():
    print("color FEN envoyée avec succès")
    return jsonify(color_FEN)

#Routes pour envoyer et recevoir le board
@app.route('/set-board-FEN', methods=['POST'])
def set_board_fen():
    global board_FEN, valhalla_FEN
    try:
        data = request.get_json()
        if "board_FEN" not in data:
            return jsonify({"error": "Key 'board_FEN' is missing"}), 400
        if "valhalla" not in data:
            return jsonify({"error": "Key 'valhalla' is missing"}), 400
    
        board_FEN = data["board_FEN"]
        valhalla_FEN = data["valhalla"]

        print("Payload reçu par Flask :")
        print("Board FEN :", board_FEN)
        print("Valhalla Cemetery :", valhalla_FEN)


        return jsonify({"status": "success", "received": board_FEN}), 200 
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get-board-FEN', methods=['GET'])
def get_board_fen():
    global board_FEN, valhalla_FEN 
    try:
        if 'board_FEN' not in globals() or board_FEN is None:
            return jsonify({"error": "No board_FEN available"}), 501
        if 'valhalla_FEN' not in globals() or valhalla_FEN is None:
            return jsonify({"error": "No valhalla data available"}), 501
        
        print("Board et Valhalla envoyés avec succès")
        print("Valhalla : " + valhalla_FEN)
        return jsonify({"status": "success", "board_FEN": board_FEN, "valhalla_FEN" : valhalla_FEN}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

# Routes pour savoir quand est-ce que le joueur humain a joué

@app.route('/set-have-played', methods=['POST'])
def set_have_played():
    global have_played_status, last_click_time
    try:
        data = request.get_json()  # Récupérer les données JSON
        have_played = data.get('have_played')  # Extraire la valeur

        if isinstance(have_played, bool):  # Vérifier que c'est bien un booléen
            current_time = time.time()
            have_played_status = have_played
            if current_time - last_click_time >= 2:
                last_click_time = current_time
                return jsonify({"message": "Statut de have_played mis à jour"}), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/get-have-played', methods=['GET'])
def get_have_played():
    def wait_for_action():
        global have_played_status
        while not have_played_status:
            time.sleep(0.1)  # Attendre 100 ms avant de vérifier à nouveau
        have_played_status = False  # Réinitialiser après l'action
        yield jsonify({"have_played": True}).data  # Retourner la réponse JSON

    return Response(stream_with_context(wait_for_action()), content_type='application/json')

@app.route('/reset-have-played', methods=['POST'])
def reset_have_played():
    global have_played_status
    have_played_status = False  # Réinitialiser à False
    return jsonify({"message": "Statut have_played réinitialisé"}), 200

@app.route('/set-state', methods=['POST'])
def set_state():
    global state
    try:
        state = request.get_json()

        return jsonify({"status" : "success"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get-state', methods=['GET'])
def get_state():
    global state 
    try:
        if 'state' not in globals() or state is None:
            return jsonify({"error": "No state available"}), 501
        
        return jsonify(state), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


    
if __name__ == '__main__':
    new_game()
    app.run(debug=True)
    

