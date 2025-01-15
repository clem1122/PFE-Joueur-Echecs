# MAIN FONCTION 
import cv2
import os
import numpy as np

from calibration import calibrate_corners, compute_transformation, rectify_image
#from image_processing import detect_differences, analyze_squares, determine_movement
from processing import detect_differences, analyze_squares, determine_movement

# --- Paramètres 
calibration_file = "chessboard_calibration.pkl"
reference_image_path = "img0.png"
output_size = (800, 800)
square_size = output_size[0] // 8
sensitivity_threshold = 20 # Seuil pour la diff de pixels 
percentage_threshold = 20 # Seuil de pourcentage de diff pour considerer une case comme modifiee

reference_image = cv2.imread(reference_image_path)
#print(reference_image.height, reference_image.width)
# cv2.imshow('reference_image', reference_image)

## Redressement de l'image de ref
# Utilisation des fonctions de calibration
input_points = calibrate_corners(calibration_file, reference_image_path, output_size)
tform = compute_transformation(input_points, output_size)
# Redresser l'image de référence
rectified_reference = rectify_image(reference_image, tform, output_size)
cv2.imshow('reference_image', rectified_reference)

# Dictionnaire des coordonnées des cases
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
# Trier par ordre numerique que les fichiers .png
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(".png")])

###############################################################################################
### MAIN LOOP 
###############################################################################################
for i in range(len(image_files) - 1):
    # Charger les images en niveau de gris
    img1 = cv2.imread(os.path.join(image_folder, image_files[i]), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(os.path.join(image_folder, image_files[i + 1]), cv2.IMREAD_GRAYSCALE)

    #Redresser les images en utilisant l'image ref
    rectified_img1 = rectify_image(img1, tform, output_size)
    rectified_img2 = rectify_image(img2, tform, output_size)

    # Calculer les diff en utilisant la fonction de image_processing
    filtered_diff = detect_differences(rectified_img1, rectified_img2, sensitivity_threshold)
    modified_cases = analyze_squares(filtered_diff, cases, square_size)
    print(f"\nTop 2 modified cases between {image_files[i]} and {image_files[i+1]}: {modified_cases}")
        # Affichage des pourcentages pour plus de détails
    for case, percentage_diff in modified_cases:
        print(f"{case} : {percentage_diff:.2f}% difference ")

    if len(modified_cases) >= 2:
        # Déterminer le mouvement
        movement_info = determine_movement(modified_cases, rectified_img1, rectified_img2, rectified_reference, cases, sensitivity_threshold)
        if "error" not in movement_info:
            print(f"Déplacement détecté : {movement_info['direction']} - Type : {movement_info['move_type']}")
            print(f"De {movement_info['start']} à {movement_info['end']}")
        else:
            print("Erreur : Pas assez de cases pour un mouvement valide.")
    else:
        print("Aucun mouvement significatif détecté.")

'''
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

        # Movment simple ou capture ?
        movement = determine_movement(modified_cases, rectified_img1, rectified_img1, rectified_reference, cases, sensitivity_threshold)

        # print le move 
        print(f"Mouvement: {movement['move_type']} de {movement['start_case']} a {movement['end_case']}")
'''