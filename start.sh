#!/bin/bash

cd PFE-Joueur-Echecs

python3 Scripts/bouton.py &  
python3 Interface/app.py
python3 -m http.server 5000 &  
python3 main.py &
