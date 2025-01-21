import cv2
import os
import numpy as np

from calibration import calibrate_corners, compute_transformation, rectify_image
from processing import (
    detect_differences, analyze_squares, determine_movement_direction, 
    is_capture, determine_piece_color, check_color,
    is_roque, is_en_passant
)
 
def oracle(img1,img2, reference_image, debug = True):

    # ------------- PARAMETERS -------------------
    threshold_diff = 30 # pour 'detect_difference' : Seuil pour la diff de pixels 
    threshold_empty = 20 #pour  'is square_empty': Seuil pour diff entre case et case empty
    # ----------------------------------------------------------------------------------------------
    calibration_file = "chessboard_calibration.pkl"
    output_size = (800, 800) # 
    square_size = output_size[0] // 8

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

    # Calibration de l'echiquier
    #reference_image = cv2.imread(reference_image)
    input_points = calibrate_corners(calibration_file, reference_image, output_size)
    tform = compute_transformation(input_points, output_size)

    # Redresser l'image de référence
    rectified_reference = rectify_image(reference_image, tform, output_size)
    rectified_reference_gray = cv2.cvtColor(rectified_reference, cv2.COLOR_BGR2GRAY)

    # Convertir les images en niveaux de gris
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    #Redresser les images en utilisant l'image ref (tform)
    rectified_img1 = rectify_image(img1, tform, output_size)
    rectified_img2 = rectify_image(img2, tform, output_size)

    # Calculer les diff en utilisant la fonction de image_processing
    filtered_diff = detect_differences(rectified_img1, rectified_img2, threshold_diff, debug)
    modified_cases = analyze_squares(filtered_diff, cases, square_size, debug)

    # ---------------------------------------------------------------------
    # Déterminer le sens du mouvement
    if len(modified_cases) >= 2:
        top_cases = [modified_cases[0], modified_cases[1]]
        origin, destination = determine_movement_direction(rectified_img1, rectified_img2, rectified_reference_gray, cases, top_cases, threshold_empty, debug)
        #print(f"\nDetected movement: {origin} -> {destination}")
    else:
        print("Errror determining mouvement: not enough modified cases.")

    # ----------------------------------------------------------------------
    # Déterminer si le mouvement est une capture
    destination_coords = cases[destination]
    capture_detected = is_capture(rectified_img1, rectified_reference_gray, destination_coords, threshold_diff, debug)
    if capture_detected:
        move_type = "CAPTURE"
        #print(f"The move is a : CAPTURE")
    else:
        move_type = "SIMPLE"
        #print(f"The move is a : SIMPLE MOVE")

    # ----------------------------------------------------------------------
    # Determiner la couleur de la piece bougee
    origin_coords = cases[origin]
    circle_mean_intensity = check_color(rectified_img1, origin_coords)
    color = determine_piece_color(circle_mean_intensity)
    

   # ----------------------------------------------------------------------
   # ------------------ CHECK FOR COUPS SPECIAUX --------------------------
   # ----------------------------------------------------------------------
   
   # -------------------
   # ------ROQUE -------
   # -------------------
    top_4_cases = [modified_cases[0][0], modified_cases[1][0], modified_cases[2][0], modified_cases[3][0]]
    roque = is_roque(top_4_cases, debug)

    # Si un roque is detected
    if roque is not None:
        move_type, color, origin, destination = is_roque(top_4_cases, debug)
    else:
        pass

   # -------------------
   # ----EN-PASSANT ----
   # -------------------
    top_3_cases = [modified_cases[0], modified_cases[1], modified_cases[2]]
    en_passant, new_origin = is_en_passant(top_3_cases, threshold_diff,debug)

    if en_passant :
        origin = new_origin
        move_type = 'EN PASSANT' 
    else:
        pass


   # -------------------
   # ----PROMOTION -----
   # -------------------


# ------------------------------------------------
    print("-------------------------------------------------------------------")
    print(f"Origin: {origin}, Destination: {destination}, Move Type: {move_type}, Piece Color: {color}")
    print("-------------------------------------------------------------------")

    return origin, destination, move_type, color

# Example usage:
def main():

    #Load empty checkboard
    reference_image = cv2.imread("Vision/photos_test/img0.png", cv2.IMREAD_COLOR)
    # Load example images
    img1 = cv2.imread("Vision/photos_en_passant/pose1.png", cv2.IMREAD_COLOR)
    img2 = cv2.imread("Vision/photos_en_passant/pose2.png", cv2.IMREAD_COLOR)

    # Process the move
    origin, destination, move_type, color = oracle(img1, img2, reference_image)

if __name__ == "__main__":
    main()