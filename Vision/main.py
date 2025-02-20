import cv2
import os
import numpy as np

from calibration import calibrate_corners, compute_transformation, rectify_image
from processing import detect_differences, analyze_squares, determine_movement_direction, is_capture, determine_piece_color, check_color

# --------------------------------------------
# ------------- PARAMETERS -------------------
# --------------------------------------------
reference_image_path = "empty.png" # Empty checkboard image
image_folder = "photos4"
sensitivity_threshold = 20 # Seuil pour la diff de pixels 
percentage_threshold = 20 # Seuil de pourcentage de diff pour considerer une case comme modifiee

# ----------------------------------------------------------------------------------------------
calibration_file = "chessboard_calibration.pkl"
output_size = (800, 800) # A
square_size = output_size[0] // 8
reference_image = cv2.imread(reference_image_path)

# Calibration de l'echiquier
input_points = calibrate_corners(calibration_file, reference_image_path, output_size)
tform = compute_transformation(input_points, output_size)

# Redresser l'image de référence
rectified_reference = rectify_image(reference_image, tform, output_size)
rectified_reference_gray = cv2.cvtColor(rectified_reference, cv2.COLOR_BGR2GRAY)

# Trier par ordre numerique que les fichiers .png du image_folder
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(".png")])

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


# ----------------------------------------------------------------------------------------
# ----------------------------------- MAIN LOOP ------------------------------------------
# ----------------------------------------------------------------------------------------

# Traite toute les images presente dans image_folder
for i in range(len(image_files) - 1):

    # Charger les images en couleur
    img1_color = cv2.imread(os.path.join(image_folder, image_files[i]))
    img2_color = cv2.imread(os.path.join(image_folder, image_files[i + 1]))
    # Convertir les images en niveaux de gris
    img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)

    #Redresser les images en utilisant l'image ref (tform)
    rectified_img1 = rectify_image(img1, tform, output_size)
    rectified_img2 = rectify_image(img2, tform, output_size)

    # Calculer les diff en utilisant la fonction de image_processing
    filtered_diff = detect_differences(rectified_img1, rectified_img2, sensitivity_threshold)
    modified_cases = analyze_squares(filtered_diff, cases, square_size)

    # print(f"\nTop 2 modified cases between {image_files[i]} and {image_files[i+1]}: {modified_cases}")
    #     # Affichage des pourcentages pour plus de détails
    
    print("\n")
    print("------------------------------")
    print("The two modified cases are: ")
    print("-", modified_cases[0][0], ",", modified_cases[0][1], "%")
    print("-", modified_cases[1][0], ",",  modified_cases[1][1], "%")

    # ---------------------------------------------------------------------
    # Déterminer le sens du mouvement
    if len(modified_cases) == 2:
        top_cases = [modified_cases[0], modified_cases[1]]
        origin, destination = determine_movement_direction(rectified_img1, rectified_img2, rectified_reference_gray, cases, top_cases)
        print(f"\nDetected movement: {origin} -> {destination}")
    else:
        print("Errror determining movment: not enough modified cases.")

    # ----------------------------------------------------------------------
    # Déterminer si le mouvement est une capture
    destination_coords = cases[destination]
    capture_detected = is_capture(rectified_img1, rectified_reference_gray, destination_coords, sensitivity_threshold)
    if capture_detected:
        print(f"The move is a : CAPTURE")
    else:
        print(f"The move is a : SIMPLE MOVE")
 
    # ----------------------------------------------------------------------
    # Determiner la couleur de la piece bougee
    origin_coords = cases[origin]
    circle_mean_intensity = check_color(rectified_img1, origin_coords)
    piece_color = determine_piece_color(circle_mean_intensity)
    print(f"\nThe piece is {piece_color}.")

    print("------------------------------")