'''
-- processing.py --
Ce fichier comporte toutes les fonctions utiles a oracle_functionet check_valhalla
'''

import cv2
import numpy as np

################################################
######### Trouver les cases modifiees ##########
################################################

# Difference filtree entre deux images
def detect_differences(img1, img2, sensitivity_threshold, debug):
    """
    Analyse les différences entre deux images avec un seuil applique
    """
    diff = cv2.absdiff(img1, img2)
    # Filtre pour ignorer les diff en dessous du threshold
    _, filtered_diff = cv2.threshold(diff, sensitivity_threshold, 255, cv2.THRESH_BINARY)
    
    if debug:
        # Verifier que l'echiquier n'as pas bouge entre deux prises
        cv2.imshow("Diff brute", diff)
        cv2.imshow("Diff filtree", filtered_diff)
        cv2.waitKey(0)  # Attendre une touche pour avancer
        cv2.destroyAllWindows()  # Fermer toutes les fenêtres après visualisation

    return filtered_diff

# Analyse repere cases modifiees dans l'echiquier
def analyze_squares(filtered_diff, cases, square_size, debug):
    """
    Analyse les différences dans chaque case en ne tenant compte que d'un cercle centré
    """
    modified_cases = []  # Liste vide

    if debug:
        print("\n-----LISTE CASES MODIFIEES:-----")

    # For toutes les cases de l'echiquier 
    for case_name, coords in cases.items():
        # Extraire les coordonnées de la case
        x_start, x_end, y_start, y_end = coords

        # Dimensions de la case
        square_width = x_end - x_start
        square_height = y_end - y_start
        center_x = x_start + square_width // 2
        center_y = y_start + square_height // 2
        radius = square_width // 4  # Rayon du cercle = largeur de la case / 4

        # Créer un masque circulaire
        mask = np.zeros_like(filtered_diff, dtype=np.uint8)
        cv2.circle(mask, (center_x, center_y), radius, 255, -1)

        # Appliquer le masque pour extraire les pixels du cercle
        masked_diff = cv2.bitwise_and(filtered_diff, filtered_diff, mask=mask)
        if debug and False:
            cv2.imshow('masked_diff', masked_diff)
            cv2.waitKey(0)
            cv2.destroyWindow('masked_diff')

        # Calculer les pourcentages de pixels différents dans le cercle
        diff_pixels = np.sum(masked_diff > 0)
        circle_area = np.pi * (radius ** 2)  # Aire du cercle
        percentage_diff = int((diff_pixels / circle_area) * 100)

        # Ajouter la case et son pourcentage de différence à la liste
        modified_cases.append((case_name, percentage_diff))

        # DEBUG
        if debug:
            print(f"Case: {case_name}, Pixels modifies: {diff_pixels}, Pourcentage: {percentage_diff}%")

    # Trier les cases par pourcentage décroissant et retourner les deux premières
    modified_cases.sort(key=lambda x: x[1], reverse=True)

    if debug:
        print(f"\nTOP 2 : {modified_cases[:2]} \n")
        #print(f"\nTOP 4 CASES MODIFIEES: {modified_cases[:4]} \n")

    return modified_cases[:4]

###############################################
####### Determiner la direction du coup #######
###############################################

# ------------- Variance ------------
def masked_variance(img, debug = False):
    # Dimensions de la case
    square_height, square_width = img.shape
    center_x = square_width // 2
    center_y = square_height // 2
    radius = square_width // 4  # Rayon du cercle = largeur de la case / 4

    # Créer un masque circulaire
    mask = np.zeros_like(img, dtype=np.uint8)
    cv2.circle(mask, (center_x, center_y), radius, 255, -1)

    # Appliquer le masque sur les deux images (case actuelle et case vide)
    masked_square_img = cv2.bitwise_and(img, img, mask=mask)

    # Extraire les pixels non nuls dans la région masquée
    masked_pixels = masked_square_img[masked_square_img > 0]

    # Calculer et retourner la variance dans la région masquée
    return np.var(masked_pixels)

# ------------ Contour --------------
def detect_contours_metric(img):
    """
    Calcule une métrique basée sur les contours pour une image donnée
    """
    # Conversion en niveaux de gris
    gray = img

    # Détection des contours avec un filtre de Sobel
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    # Binarisation pour extraire les contours
    S_, binary = cv2.threshold(sobel_magnitude, 20, 255, cv2.THRESH_BINARY)

    # Option : utiliser Canny à la place de sobel
    #binary = cv2.Canny(gray, 50, 150)

    # Calculer le nombre total de pixels de contour
    contour_metric = np.sum(binary > 0)
    return contour_metric

# Double check Variance Contour
def determine_movement_direction_with_contours(img2, cases, top_modified_cases, debug):
    """
    Détermine le mouvement en utilisant les contours pour valider la variance.
    Si les résultats sont incohérents, la méthode avec le plus grand écart relatif est priorisée.
    """
    if len(top_modified_cases) < 2:
        return None, "Erreur 'determine_movement': moins de deux cases modifiées."

    case1, case2 = top_modified_cases[0][0], top_modified_cases[1][0]

    # Extraire les coordonnées des cases
    coords1 = cases[case1]
    coords2 = cases[case2]

    # Découper les régions correspondantes
    x1_start, x1_end, y1_start, y1_end = coords1
    x2_start, x2_end, y2_start, y2_end = coords2

    # Extraction des images des cases dans img2
    square1_img2 = img2[y1_start:y1_end, x1_start:x1_end]
    square2_img2 = img2[y2_start:y2_end, x2_start:x2_end]

    # Calcul des variances
    var_case1 = masked_variance(square1_img2)
    var_case2 = masked_variance(square2_img2)

    # Calcul des métriques de contours
    contour_case1 = detect_contours_metric(square1_img2)
    contour_case2 = detect_contours_metric(square2_img2)

    # Origine est déterminée par la variance minimale
    origin_variance = case1 if var_case1 == min(var_case1, var_case2) else case2
    destination_variance = case1 if origin_variance == case2 else case2

    # Vérification avec les contours : moins de contours -> case vide
    origin_contour = case1 if contour_case1 == min(contour_case1, contour_case2) else case2
    destination_contour = case1 if origin_contour == case2 else case2

    # Validation croisée
    if debug:
        print("DETERMINE DIRECTION:")
        print('Var case 1:', int(var_case1), 'Contour case 1:', contour_case1)
        print('Var case 2:', int(var_case2), 'Contour case 2:', contour_case2)

    # En cas de divergence
    if origin_variance != origin_contour:
        if debug:
            print("WARNING: Incoherence entre variance et contours!")

        # Calcul des facteurs d'écart relatifs
        factor_var = abs(var_case1 - var_case2) / min(var_case1, var_case2)
        factor_contour = abs(contour_case1 - contour_case2) / min(contour_case1, contour_case2)
        
        if debug:
            print("Facteur var = ", factor_var, "Facteur contour= ",factor_contour)

        # Décider quelle méthode suivre selon le facteur d'écart
        if factor_var > factor_contour:
            origin = origin_variance
            destination = destination_variance
            if debug:
                print("Variance priorisee -> écart plus grand")
        else:
            origin = origin_contour
            destination = destination_contour
            if debug:
                print("Contours priorisee -> écart plus grand")

    else:
        # Si les deux méthodes sont cohérentes
        origin = origin_variance
        destination = destination_variance
        if debug:
            print("Coherence entre variance et contours.")

    if debug:
        print('Origin:', origin, 'Destination:', destination)

    return origin, destination

###############################################
################ COUPS SPECIAUX ###############
###############################################

# ----------------- ROQUE ---------------------
def is_roque(top_4_cases, debug):
    """"
    Regarder les 4 cases modifiees et determiner si elles forment une sequence du roque"
    """
    if debug :
        print('\nCHECK ROQUE:')
        print('TOP4:', top_4_cases)
    
    # Définition des séquences attendues
    petit_roque_noir = ['E8', 'G8', 'H8', 'F8']
    grand_roque_noir = ['E8', 'C8', 'A8', 'D8']
    # Définition des séquences attendues
    petit_roque_blanc = ['E1', 'G1', 'H1', 'F1']
    grand_roque_blanc = ['E1', 'C1', 'A1', 'D1']
    
    # Vérification si les cases correspondent à l'une des séquences roque
    if sorted(top_4_cases) == sorted(petit_roque_noir):
        roque_type = 'ROQUE - petit'
        roque_color = 'black'
        origin = 'E8'
        destination = 'G8'
        if debug:
            print("PETIT ROQUE NOIR detected")
    elif sorted(top_4_cases) == sorted(grand_roque_noir):
        roque_type = 'ROQUE - grand'
        roque_color = 'black'
        origin = 'E8'
        destination = 'C8'
        if debug:
            print("GRAND ROQUE NOIR detected")
    elif sorted(top_4_cases) == sorted(petit_roque_blanc):
        roque_type = 'ROQUE - petit'
        roque_color = 'white'
        origin = 'E1'
        destination = 'G1'
        if debug:
            print("PETIT ROQUE BLANC detected")
    elif sorted(top_4_cases) == sorted(grand_roque_blanc):
        roque_type = 'ROQUE - grand'
        roque_color = 'white'
        origin = 'E1'
        destination = 'C1'
        if debug:
            print("GRAND ROQUE BLANC detected")
    else:
        if debug:
            print("=> NO ROQUE")
        return None
    
    return (origin, destination)

# ----------------- EN PASSANT ----------------
def is_en_passant(top_4_cases, threshold, debug=False):
    """
    Vérifie si une prise en passant a eu lieu.
    On check si 3 cases sont modifies au dessus d'un seuil
    On check si parmis ces 3 cases, il y a lignes 5->6 ou 4->3
    """
    # Filtrer les cases qui dépassent le seuil
    valid_cases = [(case, int(case[1])) for case, pourcentage in top_4_cases if pourcentage >= threshold]
    
    if debug:
        print('\nEN PASSANT :')
        print(f"Cases valides apres filtrage ({threshold}%): {valid_cases}")
    
    # Vérifier qu'il y a au moins 3 cases valides
    if len(valid_cases) < 3:
        if debug:
            print("Moins de 3 cases depassent le seuil \n=> NO prise en passant.")
        return False, "A1", "A1"
    
    # Check si il y a redondance de lignes et de colonnes dans les 3 cases
    # lignes et colonnes extraites des cases
    lignes = [int(case[1]) for case, _ in valid_cases]
    colonnes = [case[0] for case, _ in valid_cases]

    # Vérifier les redondances dans les lignes
    lignes_redondantes = any(lignes.count(ligne) > 1 for ligne in lignes)
    # Vérifier les redondances dans les colonnes
    colonnes_redondantes = any(colonnes.count(colonne) > 1 for colonne in colonnes)

    if debug:
        print("Lignes extraites:", lignes)
        print("Colonnes extraites:", colonnes)
        print(f"Redondances - Lignes: {lignes_redondantes}, Colonnes: {colonnes_redondantes}")
    
    # Continuer seulement s'il y a redondances de lignes ET de colonnes
    if not (lignes_redondantes and colonnes_redondantes):
        if debug:
            print("Pas assez de redondances")
            print("=> NO prise en passant")
        return False, "A1", "A1"
    
    # ----Determiner case d'origine ---    
    # Vérifier si deux lignes consécutives pertinentes existent
    is_valid = (5 in lignes and 6 in lignes) or (4 in lignes and 3 in lignes)
    
    # Identifier la colonne unique ("origin")
    unique_colonnes = [colonne for colonne in colonnes if colonnes.count(colonne) == 1]
    unique_lignes = [ligne for ligne in lignes if lignes.count(ligne) == 1]


    if unique_colonnes:
        # Trouver la case complète correspondant à la colonne unique
        for case, _ in valid_cases:
            if case[0] in unique_colonnes:
                origin = case
                break

    if unique_lignes:
        # Trouver la case complète correspondant à la colonne unique
        for case, ligne in valid_cases:
            if ligne in unique_lignes:
                destination = case
                break
    
    if debug:
        print(f"Colonne unique: {unique_colonnes}, Case d'origine: {origin}")
        print(f"Ligne unique: {unique_lignes}, Case de destination: {destination}")

        print(f"Prise en passant detected: {is_valid}\n")
    
    return is_valid, origin, destination

#---------------- PROMOTION - VALHALLA ----------------
def is_case_empty(img, empty_valhalla, coords, threshold, debug):
    """
    Vérifie si une case est vide en comparant sa variance à celle de la case vide de référence.
    """
    x_start, x_end, y_start, y_end = coords

    # Extraire la région d'intérêt pour les deux images
    img_case = img[y_start:y_end, x_start:x_end]
    empty_case = empty_valhalla[y_start:y_end, x_start:x_end]

    # Calculer les variances
    var_case1 = masked_variance(img_case)
    var_case2 = masked_variance(empty_case)

    if debug:
        print('\nPROMOTION-VALHALLA:')
        print(f"Case coords: {coords}")
        print(f"Variance case img: {var_case1}, Variance case ref: {var_case2}")

    # Comparer les variances
    if abs(var_case1 - var_case2) < threshold:
        if debug:
            print("Variance proche => case vide.")
        return True
    else:
        if debug:
            print("Variance differente => case pleine.")
        return False

# ------------------------------------------------------------------------------------------------
# -------------------------------- FUNCTIONS GRAVEYARD (RIP) ------------------------------------
# ------------------------------------------------------------------------------------------------

# ----------------------------------------------
# ------------ NORMALIZATION HSV  --------------
# ----------------------------------------------

# def normalize_hsv_global(img, global_means, global_stds):
#     """
#     Normalise une image en fonction des moyennes et écarts-types globaux (par canal HSV).
#     """
#     # Convertir l'image en HSV
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

#     # Normaliser chaque canal (H, S, V)
#     for i in range(3):
#         channel = hsv[:, :, i]
#         mean_img = np.mean(channel)
#         std_img = np.std(channel)

#         # Transformation linéaire pour correspondre aux stats globales
#         hsv[:, :, i] = (channel - mean_img) * (global_stds[i] / (std_img + 1e-6)) + global_means[i]

#         # Clipping des valeurs pour rester dans les plages HSV valides
#         if i == 0:  # Hue (H) : 0-179
#             hsv[:, :, i] = np.clip(hsv[:, :, i], 0, 179)
#         else:  # Saturation et Value (S, V) : 0-255
#             hsv[:, :, i] = np.clip(hsv[:, :, i], 0, 255)

#     # Retourner l'image en BGR
#     return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


################################################
############## UNDISTORT FISHEYE  ##############
################################################

# def undistort_fisheye(image, K, D, balance=0.5):
#     """
#     Corrige la distorsion fisheye d'une image donnée.
#     """
#     h, w = image.shape[:2]
    
#     # Calculer la nouvelle matrice intrinsèque
#     new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w, h), np.eye(3), balance=balance)
    
#     # Générer les cartes de transformation
#     map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2)
    
#     # Appliquer la transformation
#     undistorted_image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)
    
#     return undistorted_imag

##############################################
# ############## DETECT CIRCLES ##############
##############################################
# def detect_circles_in_case(image, x_start, y_start, square_size, min_radius, max_radius):
#     """
#     Détecte les cercles dans une case spécifique de l'échiquier.
#     """
#     # Extraire la sous-image correspondant à la case
#     case_roi = image[y_start:y_start+square_size, x_start:x_start+square_size]
#     # gray = case_roi
#     # blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
#     # Détecter les cercles
#     circles = cv2.HoughCircles(
#         case_roi,
#         cv2.HOUGH_GRADIENT,
#         dp=1.0, #1.2
#         minDist=10, #10
#         param1=10, #50
#         param2=5, #30
#         minRadius=min_radius,
#         maxRadius=max_radius
#     )
    
#     # Retourner le nombre de cercles détectés
#     return 0 if circles is None else len(circles[0])

# def detect_circle_differences(image1, image2, cases, min_radius, max_radius, debug):
#     """
#     Compare la présence de cercles dans les cases de deux images et retourne le pourcentage de différence.
#     """
#     modified_cases = []

#     for case_name, (x_start, x_end, y_start, y_end) in cases.items():
#         # Calculer la taille de la case si nécessaire
#         square_size = x_end - x_start

#         # Détecter les cercles dans chaque case pour les deux images
#         circles_img1 = detect_circles_in_case(image1, x_start, y_start, square_size, min_radius, max_radius)
#         circles_img2 = detect_circles_in_case(image2, x_start, y_start, square_size, min_radius, max_radius)

#         # Calculer la différence et le pourcentage
#         max_circles = max(circles_img1, circles_img2)
#         if max_circles > 0:  # Éviter une division par zéro
#             percentage_diff = int((abs(circles_img1 - circles_img2) / max_circles) * 100)
#         else:
#             percentage_diff = 0  # Si aucune pièce n'est présente dans les deux cases

#         if debug:
#             print(f"{case_name}: Img1={circles_img1}, Img2={circles_img2}, Diff={percentage_diff}%")

#         # Ajouter à la liste si différence détectée
#         if percentage_diff > 0:
#             modified_cases.append((case_name, percentage_diff))

#     # Trier par pourcentage décroissant
#     modified_cases.sort(key=lambda x: x[1], reverse=True)


#     if debug:
#         #print(f"\nTOP 2 CASES MODIFIEES: {modified_cases[:2]} \n")
#         print(f"\nTOP 4 CASES MODIFIEES (cercles): {modified_cases[:4]} \n") 

#     return modified_cases


###############################################
# Determiner la couleur de la piece jouee #####
###############################################
# Verifions si la couleur de la case d'arriver differe de la case d'origine
# on check la couleur presente dans un cercle de d=l/2 centre au milieu de la case
# def check_color(img, coords):
#     """
#     Analyse la couleur moyenne dans un cercle centré dans une case.
#     img : Image en niveaux de gris.
#     coords : Coordonnées de la case (x_start, x_end, y_start, y_end).
#     """
#     x_start, x_end, y_start, y_end = coords

#     # Dimensions de la case
#     square_width = x_end - x_start
#     square_height = y_end - y_start
#     center_x = x_start + square_width // 2
#     center_y = y_start + square_height // 2
#     radius = square_width // 4  # Diamètre = largeur / 2, donc rayon = largeur / 4

#     # Créer un masque circulaire
#     mask = np.zeros_like(img, dtype=np.uint8)
#     cv2.circle(mask, (center_x, center_y), radius, 255, -1)

#     # Appliquer le masque pour extraire les pixels du cercle
#     masked_img = cv2.bitwise_and(img, img, mask=mask)

#     # Calculer l'intensité moyenne dans le cercle
#     circle_mean_intensity = cv2.mean(masked_img, mask=mask)[0]  # Moyenne des pixels dans le cercle
    
#     #print("couleur =:", circle_mean_intensity)
#     return circle_mean_intensity

# def determine_piece_color(circle_mean_intensity, threshold=50):
#     """
#     Détermine si la pièce est blanche ou noire en fonction de l'intensité moyenne.
#     atours de 255 = banc
#     proche de 0 = noir
#     """
#     return "white" if circle_mean_intensity > threshold else "black"


### Mouvement Direction

# def is_square_empty(square_img, empty_square_img, threshold, debug):
#     """
#     Détermine si une case est vide en comparant son contenu avec une image de référence vide,
#     en analysant uniquement un cercle centré dans la case.
#     True si la case est vide, False sinon.
#     """
#     # Dimensions de la case
#     square_height, square_width = square_img.shape
#     center_x = square_width // 2
#     center_y = square_height // 2
#     radius = square_width // 4  # Rayon du cercle = largeur de la case / 4

#     # Créer un masque circulaire
#     mask = np.zeros_like(square_img, dtype=np.uint8)
#     cv2.circle(mask, (center_x, center_y), radius, 255, -1)

#     # Appliquer le masque sur les deux images (case actuelle et case vide)
#     masked_square_img = cv2.bitwise_and(square_img, square_img, mask=mask)
#     masked_empty_square_img = cv2.bitwise_and(empty_square_img, empty_square_img, mask=mask)

#     var1 = np.var(masked_square_img[masked_square_img > 0])
#     var2 = np.var(masked_empty_square_img[masked_empty_square_img > 0])

#     #print('var pleine : ' + str(var1) + '\n' + "Var vide : " + str(var2))
#     # Calculer la différence moyenne dans le cercle
#     diff = cv2.absdiff(masked_square_img, masked_empty_square_img)
#     diff_value = np.mean(diff)

#     if debug:
#         print(f"Difference moyenne dans le cercle: {diff_value}, Seuil: {threshold}")
        
#     # Retourne True si la différence moyenne est inférieure au seuil
#     return diff_value #< threshold

# def determine_movement_direction(img2, cases, top_modified_cases, debug):
#     """
#     Détermine le mouvement en comparant les cases avec l'échiquier vide.
#     LA CASE QUI EST DEVENUE VIDE DANS IMG2 EST LA CASE DE DEPART

#     LOGIQUE:
#      - On compute les variances dans IMG2
#      - La plus petite variance EST LA CASE VIDE
#      - La case vide est l'origine, l'autre est la destination
#     """
#     if len(top_modified_cases) < 2:
#         return None, "Erreur 'determine_movement': moins de deux cases modifiées."

#     case1, case2 = top_modified_cases[0][0], top_modified_cases[1][0]

#     # Extraire les coordonnées des cases
#     coords1 = cases[case1]
#     coords2 = cases[case2]

#     # Découper les régions correspondantes
#     x1_start, x1_end, y1_start, y1_end = coords1
#     x2_start, x2_end, y2_start, y2_end = coords2

#     # Extraction coordonnes cases 1 et 2 dans img2
#     square1_img2 = img2[y1_start:y1_end, x1_start:x1_end]
#     square2_img2 = img2[y2_start:y2_end, x2_start:x2_end]

#     # Calcul des variances
#     var_case1 = masked_variance(square1_img2)
#     var_case2 = masked_variance(square2_img2)

#     # Origine est la mininum variance
#     origin = case1 if var_case1 == min(var_case1, var_case2) else case2
#     destination = case1 if origin==case2 else case2

#     if debug:
#         print("DETERMINE DIRECTION:")
#         print('Var case 1: ' + int(var_case1) + '\n' + "Var case 2: " + int(var_case2))
#         print('Origin:', origin, 'Destination:', destination)
#     return origin, destination



#### CERLCES

# def determine_movement_direction_circles(img2, cases, top_modified_cases, min_radius, max_radius, debug):
#     """
#     Détermine le mouvement en comparant les cases avec l'échiquier vide.
#     LA CASE QUI EST DEVENUE VIDE DANS IMG2 EST LA CASE DE DEPART

#     LOGIQUE:
#      - On compute le # de cerlces dans IMG2 dans les deux cases
#      - Le plus petit nombre EST LA CASE VIDE
#      - La case vide est l'origine, l'autre est la destination
#     """

#     if len(top_modified_cases) < 2:
#         return None, "Erreur 'determine_movement': moins de deux cases modifiées."

#     case1, case2 = top_modified_cases[0][0], top_modified_cases[1][0]

#     # Extraire les coordonnées des cases
#     coords1 = cases[case1]
#     coords2 = cases[case2]

#     # Découper les régions correspondantes
#     x1_start, x1_end, y1_start, y1_end = coords1
#     x2_start, x2_end, y2_start, y2_end = coords2

#     square_size = x1_end - x1_start

#     # Extraction coordonnes cases 1 et 2 dans img2
#     square1_img2 = img2[y1_start:y1_end, x1_start:x1_end]
#     square2_img2 = img2[y2_start:y2_end, x2_start:x2_end]

#     # Détecter les cercles dans chaque case pour les deux cases dans l'image
#     circles_case1 = detect_circles_in_case(img2, x1_start, y1_start, square_size, min_radius, max_radius)
#     circles_case2 = detect_circles_in_case(img2, x2_start, y2_start, square_size, min_radius, max_radius)

#     # Origine est la mininum variance
#     origin = case1 if circles_case1 == min(circles_case1, circles_case2) else case2
#     destination = case1 if origin==case2 else case2

#     if debug:
#         print("DETERMINE DIRECTION (with circles):")
#         print('Circles case 1: ' + str(circles_case1) + '\n' + "Circles case 2: " + str(circles_case2))
#         print('Origin:', origin, 'Destination:', destination)
#     return origin, destination


###############################################
# Determiner la capture ou le mouvement simple
###############################################

#------------------------------------------------
# La case destination etait-elle vide en image 1 ?
# def was_square_full(img, empty_board, coords, threshold, debug=False):
#     """
#     Détermine si une case était vide en utilisant la variance.
#     """
#     x_start, x_end, y_start, y_end = coords

#     # Extraire la case dans l'image et dans l'échiquier vide
#     square_img = img[y_start:y_end, x_start:x_end]
#     empty_square = empty_board[y_start:y_end, x_start:x_end]

#     var_case1 = masked_variance(square_img)
#     var_case2 = masked_variance(empty_square)

#     if debug:
#         print("\nDETERMINE CAPTURE:")
#         print('Var case image : ' + str(var_case1) + '\n' + "Var case empty : " + str(var_case2))

#     # Si les variances sont proches, alors les deux cases sont vides
#     # Return True si la case est pas vide
#     if abs(var_case1-var_case2) > 500:
#         if debug:
#             print('Var diff >500 => case pleine => capture')
#         return True
#     else:
#         if debug:
#             print('Var diff <500 => case vide => mouv simple')
#         return False

# def is_capture(img1, empty_board, destination_coords, threshold, debug=False):
#     """
#     Détecte si un mouvement est une capture en analysant la case de destination.
#     """
#     # Vérifier si la case de destination était pleine avant le coup
#     return was_square_full(img1, empty_board, destination_coords, threshold, debug)


# VALHALLY IS EMPTY

# def is_case_empty_contours(img, empty_valhalla, coords, threshold, debug):
#     """
#     Vérifie si une case est vide en comparant sa variance et ses contours
#     à ceux de la case vide de référence.
#     """
#     x_start, x_end, y_start, y_end = coords

#     # Extraire la région d'intérêt pour les deux images
#     img_case = img[y_start:y_end, x_start:x_end]
#     empty_case = empty_valhalla[y_start:y_end, x_start:x_end]

#     # Calcul des variances
#     var_case_img = masked_variance(img_case)
#     var_case_empty = masked_variance(empty_case)

#     # Vérification basée sur la variance
#     variance_check = abs(var_case_img - var_case_empty) < threshold

#     # Calcul des métriques de contours à l'aide de la fonction existante
#     contour_case_img = detect_contours_metric(img_case)
#     contour_case_empty = detect_contours_metric(empty_case)

#     # Vérification basée sur les contours
#     contour_threshold = 0.1 * contour_case_empty  # Exemple : tolérance de 10 %
#     contour_check = abs(contour_case_img - contour_case_empty) < contour_threshold

#     # Décision finale : variance ou contours
#     is_empty = False
#     if variance_check and contour_check:
#         is_empty = True  # Les deux méthodes confirment que la case est vide
#     elif not variance_check and not contour_check:
#         is_empty = False  # Les deux méthodes confirment que la case est pleine
#     else:
#         # En cas de divergence, choisir selon le facteur d'écart relatif
#         factor_var = abs(var_case_img - var_case_empty) / min(var_case_img, var_case_empty)
#         factor_contour = abs(contour_case_img - contour_case_empty) / min(contour_case_img, contour_case_empty)

#         if factor_var < factor_contour:
#             # Prioriser la variance
#             is_empty = variance_check
#             if debug:
#                 print("Variance priorisée (écart relatif plus grand).")
#         else:
#             # Prioriser les contours
#             is_empty = contour_check
#             if debug:
#                 print("Contours priorisés (écart relatif plus grand).")

#     if debug:
#         print('\nDOUBLE-CHECK PROMOTION-VALHALLA:')
#         print(f"Case coords: {coords}")
#         print(f"Variance case img: {var_case_img}, Variance case ref: {var_case_empty}")
#         print(f"Contours case img: {contour_case_img}, Contours case ref: {contour_case_empty}")
#         print(f"Variance Check: {variance_check}, Contour Check: {contour_check}")
#         print("Résultat final:", "Case vide" if is_empty else "Case pleine")

#     return is_empty