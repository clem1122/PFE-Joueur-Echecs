# PROCESS DES DIFF ENTRE IMAGE 2a2

import cv2
import numpy as np

def is_square_empty(square_img, empty_square_img, threshold=30):
    """
    Détermine si une case est vide en la comparant à l'échiquier vide.
    square_img : Image de la case actuelle (grayscale).
    empty_square_img : Image de la même case dans l'échiquier vide (grayscale).
    threshold : Seuil pour considérer qu'une case est vide.
    """
    diff = cv2.absdiff(square_img, empty_square_img)
    diff_value = np.mean(diff)  # Moyenne des différences
    return diff_value < threshold  # Si la différence est faible, la case est vide

def determine_movement_direction(img1, img2, empty_board, cases, top_modified_cases):
    """
    Détermine le mouvement en comparant les cases avec l'échiquier vide.
    LA CASE QUI EST DEVENUE VIDE EST LA CASE DE DEPART
    img1 : Image avant le coup (grayscale).
    img2 : Image après le coup (grayscale).
    empty_board : Image de l'échiquier vide (grayscale).
    cases : Dictionnaire des coordonnées des cases.
    top_modified_cases : Liste des deux cases les plus modifiées.
    """
    if len(top_modified_cases) != 2:
        return None, "Impossible de déterminer un mouvement : moins de deux cases modifiées."

    case1, case2 = top_modified_cases[0][0], top_modified_cases[1][0]

    # Extraire les coordonnées des cases
    coords1 = cases[case1]
    coords2 = cases[case2]

    # Découper les régions correspondantes
    x1_start, x1_end, y1_start, y1_end = coords1
    x2_start, x2_end, y2_start, y2_end = coords2

    square1_img1 = img1[y1_start:y1_end, x1_start:x1_end]
    square1_img2 = img2[y1_start:y1_end, x1_start:x1_end]
    square1_empty = empty_board[y1_start:y1_end, x1_start:x1_end]

    square2_img1 = img1[y2_start:y2_end, x2_start:x2_end]
    square2_img2 = img2[y2_start:y2_end, x2_start:x2_end]
    square2_empty = empty_board[y2_start:y2_end, x2_start:x2_end]

    # Déterminer si les cases sont vides
    square1_empty_after = is_square_empty(square1_img2, square1_empty)
    square2_empty_after = is_square_empty(square2_img2, square2_empty)

    # Identifier l'origine et la destination
    if square1_empty_after:  # Case 1 est devenue vide
        origin = case1
        destination = case2
    elif square2_empty_after:  # Case 2 est devenue vide
        origin = case2
        destination = case1
    else:
        return None, "Ambiguïté : aucune case ne semble être devenue vide."

    return origin, destination

#################
# Difference absolue entre deux images
def detect_differences(img1, img2, sensitivity_threshold):
    diff = cv2.absdiff(img1, img2)
    # Filtre pour ignorer les diff en dessous du threshold choisi
    _, filtered_diff = cv2.threshold(diff, sensitivity_threshold, 255, cv2.THRESH_BINARY)
    return filtered_diff

# Analyse repere cases modifiees
def analyze_squares(filtered_diff, cases, square_size):

    modified_cases = [] # Liste vide

    # Parcourir les cases du dictionnaire cases
    for case_name, coords in cases.items():
        # Voir si il y des diff dans la case actuelle
        x_start, x_end, y_start, y_end = coords
        square_diff = filtered_diff[y_start:y_end, x_start:x_end]

        # Calculer les pourcentages de pixels differents
        diff_pixels = np.sum(square_diff > 0)
        total_pixels = square_size ** 2
        percentage_diff = int((diff_pixels / total_pixels) * 100)

        # Application du seuil de difference pour considerer la case modifiee
        modified_cases.append((case_name, percentage_diff))

    modified_cases.sort(key=lambda x: x[1], reverse=True)

    return modified_cases[:2]



##################################################
# Pour determiner la capture ou le mouvement simple
###################################################

#-----------------------
# La case destination etait-elle vide ou pleine dans l'image 1 ?
def was_square_full(img, empty_board, coords, threshold):
    """
    Détermine si une case était pleine en comparant avec l'échiquier vide.
    """
    x_start, x_end, y_start, y_end = coords

    # Extraire la case dans l'image et dans l'échiquier vide
    square_img = img[y_start:y_end, x_start:x_end]
    empty_square = empty_board[y_start:y_end, x_start:x_end]

    # Calculer la différence moyenne entre les deux
    diff = np.mean(cv2.absdiff(square_img, empty_square))

    print("diff =", diff)

    # Vérifier si la différence dépasse le seuil
    return diff > threshold

def is_capture_2(img1, empty_board, destination_coords, threshold=20):
    """
    Détecte si un mouvement est une capture en analysant la case de destination.
    """
    # Vérifier si la case de destination était pleine avant le coup
    return was_square_full(img1, empty_board, destination_coords, threshold)

# Verifions si la couleur de la case d'arriver differe de la case d'origine
# on check la couleur presente dans un cercle de d=l/2 centre au milieu de la case
def check_color(img, coords):
    """
    Analyse la couleur moyenne dans un cercle centré dans une case.
    img : Image en niveaux de gris.
    coords : Coordonnées de la case (x_start, x_end, y_start, y_end).
    """
    x_start, x_end, y_start, y_end = coords

    # Dimensions de la case
    square_width = x_end - x_start
    square_height = y_end - y_start
    center_x = x_start + square_width // 2
    center_y = y_start + square_height // 2
    radius = square_width // 4  # Diamètre = largeur / 2, donc rayon = largeur / 4

    # Créer un masque circulaire
    mask = np.zeros_like(img, dtype=np.uint8)
    cv2.circle(mask, (center_x, center_y), radius, 255, -1)

    # Appliquer le masque pour extraire les pixels du cercle
    masked_img = cv2.bitwise_and(img, img, mask=mask)

    # Calculer l'intensité moyenne dans le cercle
    circle_mean_intensity = cv2.mean(masked_img, mask=mask)[0]  # Moyenne des pixels dans le cercle
    
    print("couleur =:", circle_mean_intensity)
    return circle_mean_intensity

def determine_piece_color(circle_mean_intensity, threshold=50):
    """
    Détermine si la pièce est blanche ou noire en fonction de l'intensité moyenne.
    atours de 255 = banc 
    proche de 0 = noir
    """
    return "white" if circle_mean_intensity > threshold else "black"


# ---------------------------------
# COULEURRR
# idee: analyser juste la couleur du cercle du centre de la case

def calculate_average_color(img, coords):
    """
    Calcule la couleur moyenne dans une case.
    """
    x_start, x_end, y_start, y_end = coords

    # Découper la région correspondante
    square = img[y_start:y_end, x_start:x_end]

    # Calculer la couleur moyenne (BGR)
    average_color = np.mean(square, axis=(0, 1))  # Moyenne pour chaque canal
    return average_color  # Retourne un tableau [B, G, R]

def has_color_changed(avg_color_before, avg_color_after, threshold=30):
    """
    Compare deux couleurs moyennes pour détecter un changement significatif.
    """
    # Calculer la distance euclidienne entre les deux couleurs
    color_difference = np.linalg.norm(avg_color_after - avg_color_before)

    # Vérifier si la différence dépasse le seuil
    return color_difference > threshold


def is_capture(img1, img2, destination_coords, threshold=30):
    """
    Détecte si un mouvement est une capture en analysant la couleur moyenne de la case.
    """
    # Calculer les couleurs moyennes dans la case de destination avant et après
    avg_color_before = calculate_average_color(img1, destination_coords)
    avg_color_after = calculate_average_color(img2, destination_coords)
    print("couleur moyenne before", avg_color_before)
    print("couleur moyenne after", avg_color_after)


    # Vérifier si la couleur moyenne a changé
    return has_color_changed(avg_color_before, avg_color_after, threshold)

