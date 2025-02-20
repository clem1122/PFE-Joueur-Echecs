"""
-- check_valhalla : --
input: image cimetiere, image cimetiere vide, couleur pieces bool, debug bool)
output: premiere case vide dans l'ordre de lecture du cimetiere 
Objectif: comparer l'image actuelle du cimetiere avec le cimetiere vide pour 
determiner quelle case est vide et la renvoyer (dasn l'ordre de lecture) 
"""

import cv2
from Vision.calibration import calibrate_corners, compute_transformation, rectify_image
from Vision.processing import is_case_empty

def check_valhalla(img, reference_image, isWhite, debug):
    """
    Analyse les différences entre l'image valhalla et l'image valhalla vide un   
    On veut savoir quelle case est vide (la pemiere dans l'ordre 1-20
    """

    # THRESHOLD (min diff entre la variance des images pour considerer plein )
    threshold = 200
    
    # ---------------------------------------------------------------------
    # ----------------------------- SETUP----------------------------------
    # ---------------------------------------------------------------------
    # Calibration file setup
    string = "V" if isWhite else "v"
    calibration_file = string + "_calibration.pkl"

    # Grille echiquier
    square_size = 100  # Taille d'une case carrée en pixels
    output_size = (5* square_size, 4 * square_size)  # Dimensions totales (largeur, hauteur)
    rows, cols = 5, 4  # Dimensions de la grille (lignes x colonnes)

    # Dictionnaire des coordonnées des cases
    cases = {}
    case_number = 1  # Initialisation du compteur pour les noms des cases
    for row in range(rows):
        for col in range(cols):
            x_start = col * square_size
            x_end = (col + 1) * square_size
            y_start = row * square_size
            y_end = (row + 1) * square_size
            case_name = f"{case_number}"  # Numérotation simple
            cases[case_name] = (x_start, x_end, y_start, y_end)
            case_number += 1

    # Calibration de la grille valhalla 
    input_points = calibrate_corners(calibration_file, reference_image, output_size)
    tform = compute_transformation(input_points, output_size)

    # Redresser les images
    rectified_reference = rectify_image(reference_image, tform, output_size)
    rectified_img = rectify_image(img, tform, output_size)

    # Conversion des images en niveaux de gris
    rectified_img = rectify_image(img, tform, output_size)
    rectified_reference_gray = cv2.cvtColor(rectified_reference, cv2.COLOR_BGR2GRAY)
    rectified_img_gray = cv2.cvtColor(rectified_img, cv2.COLOR_BGR2GRAY)

    # Affichage des cases en overlay sur l'image
    if debug:
        debug_image = rectified_img.copy()
        for case_name, (x_start, x_end, y_start, y_end) in cases.items():
            # Dessiner un rectangle pour chaque case
            cv2.rectangle(debug_image, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
            # Ajouter le numéro de la case
            cv2.putText(debug_image, case_name, (x_start + 5, y_start + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Afficher l'image avec les cases dessinées
        cv2.imshow("Rectified Image with Cases", debug_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # -----------------------------------------------------------------------------
    # ------------------------- TROUVER CASE VIDE ---------------------------------
    # -----------------------------------------------------------------------------
    # Prendre la premiere case vide dans l'ordre 1-20, avec la variance 
    for case_name, coords in cases.items():
        if is_case_empty(rectified_img_gray, rectified_reference_gray, coords, threshold, debug=debug):               
            if debug:
                print(f"\nLa premiere case vide est : {case_name}")
            return case_name
    
    print("ERROR: Aucune case vahalla vue comme vide")
    return None


# --------------------------
# --------- TEST -----------
# --------------------------
def main():
    #Load empty valhalla
    reference_image = cv2.imread("Images/valhalla_calibration.png", cv2.IMREAD_COLOR)
    # Load example images
    img = cv2.imread("Images/test_valhalla.png", cv2.IMREAD_COLOR)
    # Process the move
    empty_valhalla_case = check_valhalla(img, reference_image, isWhite=True, debug=True)

if __name__ == "__main__":
    main()