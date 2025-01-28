import lgpio
import time
import requests

# Ouvrir le port GPIO
handle = lgpio.gpiochip_open(0)  # Ouvre le premier GPIO chip

# Définir GPIO 2 comme une entrée (pour le bouton)
button_pin = 17
lgpio.gpio_claim_input(handle, button_pin)

# Boucle infinie pour détecter l'appui sur le bouton
try:
    while True:
        # Lire l'état du bouton
        button_state = lgpio.gpio_read(handle, button_pin)
        
        # Si le bouton est pressé (l'état est 0 pour un bouton poussoir avec pull-up)
        if button_state == 0:
            print("Bouton appuyé")
            requests.post("http://127.0.0.1/set-have-played", json={"have_played": True})
        # Attendre un court instant pour ne pas surcharger le processeur
        time.sleep(0.1)

except KeyboardInterrupt:
    print("Script interrompu.")

finally:
    # Libérer les GPIO avant de quitter
    lgpio.gpiochip_close(handle)
