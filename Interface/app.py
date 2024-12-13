from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)

CORS(app)
# Exemple de données à renvoyer
data = {
    "threat_button": "..................................111.................1.........",
    "play_button": ".................1..................1........................1..",
    "control_button": "................11111111..........1..............................",
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


if __name__ == '__main__':
    app.run(debug=True)

