# MAIN FONCTION 
import cv2
import os
import numpy as np

from calibration import calibrate_corners, compute_transformation, rectify_image
#from image_processing import detect_differences, analyze_squares, determine_movement
from processing import detect_differences, analyze_squares, determine_movement_direction, is_capture, is_capture_2, determine_piece_color, check_color

# --- Paramètres 
calibration_file = "chessboard_calibration.pkl"
reference_image_path = "empty.png"
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
rectified_reference_gray = cv2.cvtColor(rectified_reference, cv2.COLOR_BGR2GRAY)

#cv2.imshow('reference_image', rectified_reference)

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
image_folder = "photos4"
# Trier par ordre numerique que les fichiers .png
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(".png")])

###############################################################################################
### MAIN LOOP 
###############################################################################################
for i in range(len(image_files) - 1):
    # Charger les images en couleur
    img1_color = cv2.imread(os.path.join(image_folder, image_files[i]))
    img2_color = cv2.imread(os.path.join(image_folder, image_files[i + 1]))

    # Convertir les images en niveaux de gris
    img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)

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

    # Déterminer le sens du mouvement
    if len(modified_cases) == 2:
        top_cases = [modified_cases[0], modified_cases[1]]
        origin, destination = determine_movement_direction(rectified_img1, rectified_img2, rectified_reference_gray, cases, top_cases)
        print(f"Movement detected: {origin} -> {destination}")
    else:
        print("Not enough modified cases to determine movement.")


    # Déterminer si le mouvement est une capture
    destination_coords = cases[destination]
    capture_detected = is_capture_2(rectified_img1, rectified_reference_gray, destination_coords)
    if capture_detected:
        print(f"The move is a CAPTURE!")
    else:
        print(f"The move is a simple move.")
 
    origin_coords = cases[origin]
    circle_mean_intensity = check_color(rectified_img1, origin_coords)
    piece_color = determine_piece_color(circle_mean_intensity)

    print(f"The piece moving from {origin} is {piece_color}.")



    # # Déterminer si le mouvement est une capture
    # destination_coords = cases[destination]
    # capture_detected = is_capture(img1_color, img2_color, destination_coords)
    # if capture_detected:
    #     print(f"The move {origin} -> {destination} is a capture!")
    # else:
    #     print(f"The move {origin} -> {destination} is a simple move.")
