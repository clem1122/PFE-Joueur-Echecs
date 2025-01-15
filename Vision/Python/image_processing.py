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

# Determiner le mouvement effectue
def determine_movement(modified_cases, rectified_img1, rectified_img2, rectified_reference, cases, sensitivity_threshold):
    start_case, end_case = '', ''

    # Parcours les cases modifees
    for case in modified_cases:
        #Extraction coords
        coords = cases[case]
        x_start, x_end, y_start, y_end = coords

        # Extraire les régions d'intérêt de img_ref et img i, i+1
        ref_square = rectified_reference[y_start:y_end, x_start:x_end]
        prev_square = rectified_img1[y_start:y_end, x_start:x_end]
        curr_square = rectified_img2[y_start:y_end, x_start:x_end]

        # Calculer les différences de imag i, i+1 par rapport à img_ref
        diff_prev = cv2.absdiff(prev_square, ref_square)
        diff_curr = cv2.absdiff(curr_square, ref_square)

        diff_prev_count = np.sum(diff_prev > sensitivity_threshold)
        diff_curr_count = np.sum(diff_curr > sensitivity_threshold)

        # Déterminer les cases de départ et d'arrivée
        # 'si l'image precedent a - de diff que image actuelle, la piece est partie de cette case'
        if diff_prev_count > diff_curr_count:
            start_case = case
        elif diff_prev_count < diff_curr_count:
            end_case = case

    if start_case and end_case:
        # Vérifier si c'était un mouvement simple ou une capture
        #extraction zones d'intteret
        arrival_prev_square = rectified_img1[cases[end_case][2]:cases[end_case][3], cases[end_case][0]:cases[end_case][1]]
        reference_square = rectified_reference[cases[end_case][2]:cases[end_case][3], cases[end_case][0]:cases[end_case][1]]
        # Calcul diff de pixels 
        diff_prev = cv2.absdiff(arrival_prev_square, reference_square)
        diff_prev_count = np.sum(diff_prev > sensitivity_threshold)

        #Max est 100x100=10,000
        if diff_prev_count < 250:
            move_type = "simple"
            #print(f"Mouvement simple détecté: {start_case} -> {end_case}")
        else:
            move_type = "capture"
            #print(f"Capture détectée: {start_case} -> {end_case}")
    else:
        move_type = "error"
        #print("Aucun mouvement détecté.")


# New structure ##

