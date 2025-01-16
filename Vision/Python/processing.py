# PROCESS DES DIFF ENTRE IMAGE 2a2

import cv2
import numpy as np

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


# Fonctions logique pour determiner
def is_case_empty(case_coords, reference_img, sensitivity_threshold):
    """
    Vérifie si une case est vide en utilisant l'image de référence.
    """
    x_start, x_end, y_start, y_end = case_coords
    reference_square = reference_img[y_start:y_end, x_start:x_end]
    diff = np.sum(reference_square > sensitivity_threshold)
    return diff < 250  # Seuil pour considérer la case vide.

def is_case_occupied(case_coords, img, reference_img, sensitivity_threshold):
    """
    Vérifie si une case est occupée en comparant l'image actuelle avec l'image de référence.
    """
    x_start, x_end, y_start, y_end = case_coords
    current_square = img[y_start:y_end, x_start:x_end]
    reference_square = reference_img[y_start:y_end, x_start:x_end]
    diff = np.sum(cv2.absdiff(current_square, reference_square) > sensitivity_threshold)
    return diff > 250  # Seuil pour considérer la case occupée.

def did_piece_leave_case(case_coords, img_prev, reference_img, sensitivity_threshold):
    """
    Vérifie si une pièce a quitté la case.
    """
    return is_case_occupied(case_coords, img_prev, reference_img, sensitivity_threshold) and is_case_empty(case_coords, reference_img, sensitivity_threshold)

def did_piece_arrive_in_case(case_coords, img_curr, reference_img, sensitivity_threshold):
    """
    Vérifie si une pièce est arrivée sur la case.
    """
    return is_case_empty(case_coords, reference_img, sensitivity_threshold) and is_case_occupied(case_coords, img_curr, reference_img, sensitivity_threshold)

def determine_movement_info(start_case, end_case, img_prev, img_curr, reference_img, cases, sensitivity_threshold):
    """
    Détermine la direction et le type du mouvement en utilisant des fonctions booléennes.
    """
    start_coords = cases[start_case]
    end_coords = cases[end_case]

    left_start = did_piece_leave_case(start_coords, img_prev, reference_img, sensitivity_threshold)
    arrived_end = did_piece_arrive_in_case(end_coords, img_curr, reference_img, sensitivity_threshold)

    if left_start and arrived_end:
        direction = "from start to end"
        move_type = "simple" if is_case_empty(end_coords, img_prev, sensitivity_threshold) else "capture"
    elif did_piece_leave_case(end_coords, img_prev, reference_img, sensitivity_threshold) and did_piece_arrive_in_case(start_coords, img_curr, reference_img, sensitivity_threshold):
        direction = "from end to start"
        move_type = "simple" if is_case_empty(start_coords, img_prev, sensitivity_threshold) else "capture"
    else:
        direction = "unknown"
        move_type = "unknown"

    return {
        "start": start_case,
        "end": end_case,
        "direction": direction,
        "move_type": move_type
    }

def determine_movement(modified_cases, rectified_img1, rectified_img2, reference_img, cases, sensitivity_threshold):
    """
    Fonction principale appelée pour déterminer le mouvement entre deux images.
    """
    if len(modified_cases) >= 2:
        start_case, end_case = modified_cases[0][0], modified_cases[1][0]
        return determine_movement_info(start_case, end_case, rectified_img1, rectified_img2, reference_img, cases, sensitivity_threshold)
    else:
        return {
            "error": "Pas assez de cases modifiées pour déterminer un mouvement."
        }