import requests
import PChess as pc

B = pc.Board();

url = "http://127.0.0.1:5000/set-color-FEN"
payload = {"threats": B.threats(), 
           "playable": B.playable(), 
           "controlled": B.controlled()}

response = requests.post(url, json=payload)

if response.status_code == 200:
    print("OK")