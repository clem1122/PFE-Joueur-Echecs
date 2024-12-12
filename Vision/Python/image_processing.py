# PROCESS DES DIFF ENTRE IMAGE 2a2

import cv2
import numpy as np

# Difference absolue entre deux images
def detect_differences(img1, img2, sensitivity_threshold):
    diff = cv2.absdiff(img1, img2)
    # Filtre pour ignorer les diff en dessous du threshold dchoisi
    _, filtered_diff = cv2.threshold(diff, sensitivity_threshold, 255, cv2.THRESH_BINARY)
    return filtered_diff

def determine_movement(modified_cases, rectified_img1, rectified_img2, rectified_reference, cases, sensitivity_threshold):
    start_case, end_case = '', ''

    for case in modified_cases:
        coords = cases[case]
        x_start, x_end, y_start, y_end = coords

        # Extraire les régions d'intérêt
        ref_square = rectified_reference[y_start:y_end, x_start:x_end]
        prev_square = rectified_img1[y_start:y_end, x_start:x_end]
        curr_square = rectified_img2[y_start:y_end, x_start:x_end]

        # Calculer les différences par rapport à l'image de référence
        diff_prev = cv2.absdiff(prev_square, ref_square)
        diff_curr = cv2.absdiff(curr_square, ref_square)

        diff_prev_count = np.sum(diff_prev > sensitivity_threshold)
        diff_curr_count = np.sum(diff_curr > sensitivity_threshold)

        # Déterminer les cases de départ et d'arrivée
        if diff_prev_count > diff_curr_count:
            start_case = case
        elif diff_prev_count < diff_curr_count:
            end_case = case

    if start_case and end_case:
        # Vérifier si c'était un mouvement simple ou une capture
        arrival_prev_square = rectified_img1[cases[end_case][2]:cases[end_case][3], cases[end_case][0]:cases[end_case][1]]
        reference_square = rectified_reference[cases[end_case][2]:cases[end_case][3], cases[end_case][0]:cases[end_case][1]]
        diff_prev = cv2.absdiff(arrival_prev_square, reference_square)
        diff_prev_count = np.sum(diff_prev > sensitivity_threshold)

        if diff_prev_count < 250:
            print(f"Mouvement simple détecté: {start_case} -> {end_case}")
        else:
            print(f"Capture détectée: {start_case} -> {end_case}")
    else:
        print("Aucun mouvement détecté.")


# Analyse repere cases modifiees
def analyze_squares(filtered_diff, cases, percentage_threshold, square_size):
    modified_cases = [] # Liste vide

    # Parcourir les cases du dictionnaire cases
    for case_name, coords in cases.items():
        # Voir si il y des diff dans la case actuelle
        x_start, x_end, y_start, y_end = coords
        square_diff = filtered_diff[y_start:y_end, x_start:x_end]

        # Calculer les pourcentages de pixels differents
        diff_pixels = np.sum(square_diff > 0)
        total_pixels = square_size ** 2
        percentage_diff = (diff_pixels / total_pixels) * 100

        # Application du seuil de difference pour considerer la case modifiee
        if percentage_diff > percentage_threshold:
            modified_cases.append(case_name)
    return modified_cases
