from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)

CORS(app)
# Exemple de données à renvoyer
color_FEN = {
    "threats": "...............................................................",
    "playable": "................................................................",
    "controlled": ".................................................................",
    "toggle4": "Aide"
}

@app.route('/')
def home():
    print("La route / a été appelée")
    return "Le serveur flask tourne bien à cette URL"


@app.route('/get-info/<toggle_id>', methods=['GET'])
def get_info(toggle_id):
    print(toggle_id)
    # Retourne les infos pour un bouton spécifique
    if toggle_id in data:
        print(f"Returning info for {toggle_id}: {data[toggle_id]}")
        result = jsonify({"FEN": data[toggle_id]})  # Debug
        print("JSON : ", result)
        return result
    else:
        return jsonify({"error": "Bouton non trouvé"}), 404


@app.route('/set-color-FEN', methods=['GET'])
def set_color_FEN():
    global color_FEN
    data = request.get_json()
    color_FEN["controlled"] = data.controlled
    color_FEN["playable"] = data.playable
    color_FEN["threats"] = data.threats
    return jsonify({"status": "success", "received": color_FEN})


if __name__ == '__main__':
    app.run(debug=True)

