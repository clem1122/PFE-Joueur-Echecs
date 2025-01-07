import requests
import PChess as pc
import sys
import argparse

B = pc.Board()

url = "http://127.0.0.1:5000/set-color-FEN"
payload = {"threats": B.threats(True), 
           "playable": "................11.............................................", 
           "controlled": ".............................................11................"}

response = requests.post(url, json=payload)

if response.status_code == 200:
    print("OK")