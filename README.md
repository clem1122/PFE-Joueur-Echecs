# PFE-Joueur-Echecs

Auteurs :  
Luca Perrin  
Gaëlle Dumont Le Brazidec  
Maï-Anh Dang  
Clément Pénicaut  
Louis Le Berre  

Ce repository contient les codes nécessaires au fonctionnement d'un robot joueur d'échecs mis en place à Sorbonne Universités durant l'année 2024-2025.

## Matériel

- Robot Niryo Studio Ned2 (https://niryo.com/fr/produit/bras-robotise-6-axes/)
- Planche usinée permettant de contenir le robot et le plateau d'échec
- Un plateau d'échecs de dimensions 40x40
- 2 jeux de pièces d'échec
- Une raspberry
- Un écran
- Une souris

## Software

- Librairie pyniryo
- L'ensemble des codes de ce git

## Contenu du git

- Un main.py pour tout lancer
- Un backup.txt pour relancer une partie rapidement en cas de crash
- Un dossier Vision contenant toutes les fonctions permettant le fonctionnement de la caméra du robot
- Un dossier Scripts contenant toutes les fonctions permettant de faire bouger le robot. Il contient aussi un module python PChess.so disponible à l'adresse https://github.com/clem1122/PChess
- Un dossier Interface contenant toutes les fonctions permettant de dessiner l'interface
- Un dossier IA contenant l'IA du robot

## Mise en place du système

### 1. Installer le software
- Avoir une Raspberry tournant sur Linux/Ubuntu
- Dans un dossier, télécharger tous les dossiers contenus dans ce git
- S'assurer au fur et à mesure du lancement de disposer de toutes les librairies python nécessaires (listées surtout dans l'en tête du main.py)

### 2. Installer le hardware
- Installer la planche, insérer le plateau de jeu dans l'extrusion prévue à cet effet, placer le robot contre cette dernière dans l'emplacement prévu
- Brancher l'alimentation du robot et de la raspberry, lui brancher un clavier et une souris

### 3. Faire tourner les applications nécessaires

- Ouvrir un terminal, aller dans le dossier Interface et taper "python3 app.py" pour lancer la flask python qui gère l'interface
- Ouvrir un terminal, aller dans le dossier Script et taper "python3 bouton.py" (si vous avez un bouton connecté au PIN 17 de la raspberry)
- Ouvrir un terminal, aller dans le dossier Interface et taper "python3 -m http.server 5500" pour lancer un serveur sur le port 5500
- Ouvrir Firefox et taper dans la barre de recherche "127.0.0.1:5500" pour accéder à la page de l'interface

(NOTE : toutes ces étapes sont effectuées dans le ./start que l'on peut activer depuis la racine du projet, sauf que ce fichier lance la partie en plus et reste assez opaque si on ne sait pas ce qu'il fait)

### 4. Faire fonctionner et calibrer le robot
- Connecter le robot via hotspot ou wifi (via l'application NiryStudio) (noter l'IP du robot obtenue par wifi, qui vaut 10.10.10.10 si c'est en hotspot)
- Aller dans le dossier Robot.py et changer l'IP du robot pour faire correspondre
- Aller dans le terminal et taper "python3 main.py --calibration" pour calibrer le robot
- A la racine du projet, taper "python3 main.py -s"
