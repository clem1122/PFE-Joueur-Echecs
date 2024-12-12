# MAIN FONCTION 
# appelle calibration_utils et image_processing

import cv2
import os
import numpy as np
from calibration_utils import calibrate_corners, compute_transformation, rectify_image
from image_processing import detect_differences, analyze_squares, determine_movement

# Paramètres
calibration_file = "chessboard_calibration.pkl"
reference_image_path = "empty.png"
output_size = (800, 800) # Modifier et voir la difference (400,400)
square_size = output_size[0] // 8
sensitivity_threshold = 40 # Seuil pour la diff de pixels 
percentage_threshold = 22 # Seuil de pourcentage de diff pour considerer une case comme modifiee

## Redressement de l'image de ref
# Utilisation des fonctions de calibration_utils
input_points = calibrate_corners(calibration_file, reference_image_path, output_size)
tform = compute_transformation(input_points, output_size)
# Charger et redresser l'image de référence
reference_image = cv2.imread(reference_image_path)
rectified_reference = rectify_image(reference_image, tform, output_size)

# Creation du dictionnaire des coordonnées des cases
cases = {}
for row in range(8):
    for col in range(8):
        x_start = col * square_size
        x_end = (col + 1) * square_size
        y_start = (7 - row) * square_size
        y_end = (8 - row) * square_size
        case_name = f"{chr(65 + col)}{row + 1}"
        cases[case_name] = (x_start, x_end, y_start, y_end)

# Chargement des images
image_folder = "photos"
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(".png")])

# Main boucle 
for i in range(len(image_files) - 1):
    # Charger les images en niveau de gris
    img1 = cv2.imread(os.path.join(image_folder, image_files[i]), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(os.path.join(image_folder, image_files[i + 1]), cv2.IMREAD_GRAYSCALE)
    #Redresser les images en utilisant l'image ref 
    rectified_img1 = rectify_image(img1, tform, output_size)
    rectified_img2 = rectify_image(img2, tform, output_size)
    # Calculer les diff en utilisant la fonction de image_processing
    filtered_diff = detect_differences(rectified_img1, rectified_img2, sensitivity_threshold)
    modified_cases = analyze_squares(filtered_diff, cases, percentage_threshold, square_size)

    # Print diff entre les cases
    if modified_cases:
        print(f"Differences between {image_files[i]} and {image_files[i+1]}: {modified_cases}")
        for case in modified_cases:
            x_start, x_end, y_start, y_end = cases[case]
            square_diff = filtered_diff[y_start:y_end, x_start:x_end]
            diff_pixels = np.sum(square_diff > 0)
            total_pixels = square_size ** 2
            percentage_diff = int((diff_pixels / total_pixels) * 100)
            print(f"Case {case}:{percentage_diff:.2f}%")

