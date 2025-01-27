import lgpio
import time

# Ouvrir le port GPIO
handle = lgpio.gpiochip_open(0)  # Ouvre le premier GPIO chip

# Définir GPIO 2 comme une entrée (pour le bouton)
button_pin = 17
lgpio.gpio_claim_input(handle, button_pin)

# Définir GPIO 17 comme une sortie (pour la LED)
led_pin = 27
lgpio.gpio_claim_output(handle, led_pin)

# Boucle infinie pour détecter l'appui sur le bouton
try:
    while True:
        # Lire l'état du bouton
        button_state = lgpio.gpio_read(handle, button_pin)
        
        # Si le bouton est pressé (l'état est 0 pour un bouton poussoir avec pull-up)
        if button_state == 1:
            print("Bouton appuyé!")
            # Allumer la LED
            lgpio.gpio_write(handle, led_pin, 1)
        else:
            # Éteindre la LED si le bouton n'est pas pressé
            lgpio.gpio_write(handle, led_pin, 0)
        
        # Attendre un court instant pour ne pas surcharger le processeur
        time.sleep(0.1)

except KeyboardInterrupt:
    print("Script interrompu.")

finally:
    # Libérer les GPIO avant de quitter
    lgpio.gpiochip_close(handle)
