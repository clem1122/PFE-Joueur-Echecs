import cv2
import numpy as np

################################################
######### Trouver les cases modifiees ##########
################################################

# Difference entre deux images
def detect_differences(img1, img2, sensitivity_threshold, debug):
    diff = cv2.absdiff(img1, img2)
    # Filtre pour ignorer les diff en dessous du threshold
    _, filtered_diff = cv2.threshold(diff, sensitivity_threshold, 255, cv2.THRESH_BINARY)
    
    # DEBUG
    if debug:
        # Verifier que l'echiquier n'as pas bouge entre deux prises
        cv2.imshow("Diff brute", diff)
        cv2.imshow("Diff filtree", filtered_diff)
        cv2.waitKey(0)  # Attendre une touche pour avancer
        cv2.destroyAllWindows()  # Fermer toutes les fenêtres après visualisation

    return filtered_diff

# Analyse repere cases modifiees
def analyze_squares(filtered_diff, cases, square_size, debug):
    """
    Analyse les différences dans chaque case en ne tenant compte que d'un cercle centré.
    """
    modified_cases = []  # Liste vide

    # DEBUG
    if debug:
        print("\nLISTE CASES MODIFIEES:")

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
        print(f"\nTOP 2 CASES MODIFIEES: {modified_cases[:2]} \n")
        #print(f"\nTOP 4 CASES MODIFIEES: {modified_cases[:4]} \n")

    return modified_cases[:4]

###############################################
####### Determiner la direction du coup #######
###############################################

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
    return np.var(img[img > 0])

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

def determine_movement_direction(img2, cases, top_modified_cases, debug):
    """
    Détermine le mouvement en comparant les cases avec l'échiquier vide.
    LA CASE QUI EST DEVENUE VIDE DANS IMG2 EST LA CASE DE DEPART

    LOGIC:
     - On compute les variances dans IMG2
     - La plus petite variance EST LA CASE VIDE
     - La case vide est l'origine, l'autre est la destination
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

    # Extraction coordonnes cases 1 et 2 dans img2
    square1_img2 = img2[y1_start:y1_end, x1_start:x1_end]
    square2_img2 = img2[y2_start:y2_end, x2_start:x2_end]

    # Calcul des variances
    var_case1 = masked_variance(square1_img2)
    var_case2 = masked_variance(square2_img2)

    # Origine est la mininum variance
    origin = case1 if var_case1 == min(var_case1, var_case2) else case2
    destination = case1 if origin==case2 else case2

    if debug:
        print("DETERMINE DIRECTION:")
        print('Var case 1: ' + str(var_case1) + '\n' + "Var case 2: " + str(var_case2))
        print('Origin:', origin, 'Destination:', destination)
    return origin, destination

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

###############################################
################ COUPS SPECIAUX ###############
###############################################

################## ROQUE ######################
def is_roque(top_4_cases, debug):
    """"
    Regarde les 4 cases modifiees et determine si elles correspondent a la sequence du roque"
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
            print("Aucune correspondance avec un roque")
        return None
    
    return (origin, destination)


################## EN PASSANT ###################
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
            print("Moins de 3 cases depassent le seuil => NO prise en passant.")
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
            print("Pas de redondances suffisantes. Pas de prise en passant.")
        return False, "A1", "A1"
    
    # ----Determiner case d'origine ---    
    # Vérifier si deux lignes consécutives pertinentes existent
    is_valid = (5 in lignes and 6 in lignes) or (4 in lignes and 3 in lignes)
    
    # Identifier la colonne unique ("origin")
    unique_colonnes = [colonne for colonne in colonnes if colonnes.count(colonne) == 1]
    unique_lignes = [ligne for ligne in lignes if lignes.count(ligne) == 1]

    # if debug:
    #     print(f"Colonne unique: {unique_colonnes}")
    #     print(f"Ligne unique: {unique_lignes}")

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