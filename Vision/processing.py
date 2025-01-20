import cv2
import numpy as np

###############################################
######### Trouver les cases modifiees ##########
###############################################

# Difference entre deux images
def detect_differences(img1, img2, sensitivity_threshold, debug):
    diff = cv2.absdiff(img1, img2)
    # Filtre pour ignorer les diff en dessous du threshold
    _, filtered_diff = cv2.threshold(diff, sensitivity_threshold, 255, cv2.THRESH_BINARY)
    
    # DEBUG
    if debug == True :
        # Verifier que l'echiquier n'as pas bouge entre deux prises
        cv2.imshow("Diff brute", diff)
        cv2.imshow("Diff filtree", filtered_diff)
        cv2.waitKey(0)  # Attendre une touche pour avancer
        cv2.destroyAllWindows()  # Fermer toutes les fenêtres après visualisation

    return filtered_diff

# Analyse repere cases modifiees
def analyze_squares(filtered_diff, cases, square_size, debug):
    """
    Analyse les différences dans chaque case en ne tenant compte que d'un cercle centré.
    """
    modified_cases = []  # Liste vide

    # DEBUG
    if debug:
        print("\nLISTE CASES MODIFIEES:")

    for case_name, coords in cases.items():
        # Extraire les coordonnées de la case
        x_start, x_end, y_start, y_end = coords

        # Dimensions de la case
        square_width = x_end - x_start
        square_height = y_end - y_start
        center_x = x_start + square_width // 2
        center_y = y_start + square_height // 2
        radius = square_width // 4  # Rayon du cercle = largeur de la case / 4

        # Créer un masque circulaire
        mask = np.zeros_like(filtered_diff, dtype=np.uint8)
        cv2.circle(mask, (center_x, center_y), radius, 255, -1)

        # Appliquer le masque pour extraire les pixels du cercle
        masked_diff = cv2.bitwise_and(filtered_diff, filtered_diff, mask=mask)

        # Calculer les pourcentages de pixels différents dans le cercle
        diff_pixels = np.sum(masked_diff > 0)
        circle_area = np.pi * (radius ** 2)  # Aire du cercle
        percentage_diff = int((diff_pixels / circle_area) * 100)

        # Ajouter la case et son pourcentage de différence à la liste
        modified_cases.append((case_name, percentage_diff))

        # DEBUG
        if debug:
            print(f"Case: {case_name}, Pixels modifies: {diff_pixels}, Pourcentage: {percentage_diff}%")

    # Trier les cases par pourcentage décroissant et retourner les deux premières
    modified_cases.sort(key=lambda x: x[1], reverse=True)

    if debug:
        print(f"\nTOP 2 CASES MODIFIEES: {modified_cases[:2]}")

    return modified_cases[:2]

###############################################
####### Determiner la direction du coup #######
###############################################

def visualize_diff_with_highlight(img, diff, cases, highlighted_cases, debug):
    """
    Visualise l'image complète avec les différences entre l'image actuelle et l'échiquier vide,
    tout en mettant en surbrillance les cases d'intérêt.
    """
    # Convertir l'image en couleur si elle est en niveaux de gris (pour dessiner en couleur)
    if len(img.shape) == 2:
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_color = img.copy()

    # Dessiner les cases modifiées
    for case, coords in cases.items():
        x_start, x_end, y_start, y_end = coords
        color = (0, 255, 0) if case not in highlighted_cases else (0, 0, 255)  # Vert normal, Rouge pour surbrillance
        thickness = 2 if case in highlighted_cases else 1  # Épaisseur différente pour les cases d'intérêt
        cv2.rectangle(img_color, (x_start, y_start), (x_end, y_end), color, thickness)

    # Superposer la différence avec l'échiquier vide
    diff_overlay = cv2.addWeighted(img_color, 0.7, cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR), 0.3, 0)

    if debug:
        cv2.imshow("Différence avec surbrillance des cases", diff_overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return diff_overlay


def is_square_empty(square_img, empty_square_img, threshold, debug):
    """
    Détermine si une case est vide en comparant son contenu avec une image de référence vide,
    en analysant uniquement un cercle centré dans la case.
    - True si la case est vide, False sinon.
    """
    # Dimensions de la case
    square_height, square_width = square_img.shape
    center_x = square_width // 2
    center_y = square_height // 2
    radius = square_width // 4  # Rayon du cercle = largeur de la case / 4

    # Créer un masque circulaire
    mask = np.zeros_like(square_img, dtype=np.uint8)
    cv2.circle(mask, (center_x, center_y), radius, 255, -1)

    # Appliquer le masque sur les deux images (case actuelle et case vide)
    masked_square_img = cv2.bitwise_and(square_img, square_img, mask=mask)
    masked_empty_square_img = cv2.bitwise_and(empty_square_img, empty_square_img, mask=mask)

    # Calculer la différence moyenne dans le cercle
    diff = cv2.absdiff(masked_square_img, masked_empty_square_img)
    diff_value = np.mean(diff)

    if debug:
        print(f"Difference moyenne dans le cercle: {diff_value}, Seuil: {threshold}")
        # Visualisation pour debug
        # cv2.imshow("Masque circulaire sur la case actuelle", masked_square_img)
        # cv2.imshow("Masque circulaire sur la case vide", masked_empty_square_img)
        # cv2.imshow("Différence dans le cercle", diff)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    # Retourne True si la différence moyenne est inférieure au seuil
    return diff_value < threshold

def determine_movement_direction(img1, img2, empty_board, cases, top_modified_cases, threshold_empty, debug):
    """
    Détermine le mouvement en comparant les cases avec l'échiquier vide.
    LA CASE QUI EST DEVENUE VIDE DANS IMG2 EST LA CASE DE DEPART
    """
    if len(top_modified_cases) != 2:
        return None, "Erreur 'determine_movement': moins de deux cases modifiées."

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

    # Calcul de la différence complète
    diff = cv2.absdiff(img2, empty_board)
     # Visualisation de la différence avec surbrillance des cases
    if debug:
        visualize_diff_with_highlight(img2, diff, cases, [case1, case2], debug)

    # Déterminer dans image 2 quelle case est devenue vide
    square1_empty_after = is_square_empty(square1_img2, square1_empty, threshold_empty, debug)
    square2_empty_after = is_square_empty(square2_img2, square2_empty, threshold_empty, debug)

    # Identifier l'origine et la destination
    if square1_empty_after:  # Case 1 est devenue vide
        origin = case1
        destination = case2
    elif square2_empty_after:  # Case 2 est devenue vide
        origin = case2
        destination = case1
    else:
        return None, "Erreur: Aucune case ne semble etre devenue vide -> 'is square empty'."

    return origin, destination

###############################################
# Determiner la capture ou le mouvement simple
###############################################

#-----------------------
# La case destination etait-elle vide ou pleine dans l'image 1 ?
def was_square_full(img, empty_board, coords, threshold, debug):
    """
    Détermine si une case était pleine en comparant avec l'échiquier vide.
    """
    x_start, x_end, y_start, y_end = coords

    # Extraire la case dans l'image et dans l'échiquier vide
    square_img = img[y_start:y_end, x_start:x_end]
    empty_square = empty_board[y_start:y_end, x_start:x_end]

    # Calculer la différence moyenne entre les deux
    diff = np.mean(cv2.absdiff(square_img, empty_square))

    # Vérifier si la différence dépasse le seuil
    return diff > threshold

def is_capture(img1, empty_board, destination_coords, threshold, debug):
    """
    Détecte si un mouvement est une capture en analysant la case de destination.
    """
    # Vérifier si la case de destination était pleine avant le coup
    return was_square_full(img1, empty_board, destination_coords, threshold, debug)

###############################################
# Determiner la couleur de la piece jouee #####
###############################################

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
    
    #print("couleur =:", circle_mean_intensity)
    return circle_mean_intensity

def determine_piece_color(circle_mean_intensity, threshold=50):
    """
    Détermine si la pièce est blanche ou noire en fonction de l'intensité moyenne.
    atours de 255 = banc
    proche de 0 = noir
    """
    return "white" if circle_mean_intensity > threshold else "black"


###############################################
###### Verifier que la photo est valide ######
###############################################

# Verifier si la photo est valide, cad que l'echiquier est bien visible
# Cela renforce si jamais une main cache l'echiquier

