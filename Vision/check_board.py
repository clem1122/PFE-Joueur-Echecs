###############################################
###### Verifier que la photo est valide ######
###############################################

# Verifier si la photo est valide, cad que l'echiquier est bien visible
# Cela renforce si jamais une main cache l'echiquier

import cv2
import numpy as np

def is_chessboard_visible(img):
    """
    Vérifie si l'image contient un échiquier bien visible pour analyse.
    :param image_path: Chemin de l'image à analyser
    :return: Booléen indiquant si l'échiquier est bien capturé
    """
 
    # Convertir l'image en niveaux de gris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # Appliquer un flou gaussien pour réduire le bruit
    #blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Détection des coins de l'échiquier
    pattern_size = (7, 7)  # Par défaut pour un échiquier 8x8 (7 intersections par côté)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    if not ret:
        print("Échiquier non détecté.")
        return False

    # Vérification des proportions de l'échiquier
    h, w = img.shape[:2]
    margin = 50  # Tolérance pour les bordures
    if (
        corners[0][0][0] < margin or corners[0][0][1] < margin or
        corners[-1][0][0] > w - margin or corners[-1][0][1] > h - margin
    ):
        print("Échiquier mal positionné ou coupé.")
        return False

    # Dessiner les coins détectés pour validation visuelle
    debug_image = img.copy()
    cv2.drawChessboardCorners(debug_image, pattern_size, corners, ret)

    # Afficher l'image pour vérification
    cv2.imshow("Échiquier détecté", debug_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("Échiquier bien capturé.")
    return True


def detect_chessboard_grid(img):
    # Charger l'image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Détecter les contours
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Transformation de Hough pour détecter les lignes
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("Grille détectée", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("Grille détectée.")
        return True
    else:
        print("Aucune grille détectée.")
        return False

# Exemple d'utilisation
img = cv2.imread("vision/photos3/pose3.png", cv2.IMREAD_COLOR)
if img is None:
    print("Erreur: Impossible de charger l'image.")

detect_chessboard_grid(img)

#is_chessboard_visible(img)