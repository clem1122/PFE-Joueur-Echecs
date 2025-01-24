# CALIBRATION AVEC LA PHOTO EMPTY
import cv2
import numpy as np
import pickle
import os
#from Scripts.Robot import Robot
#from pyniryo.vision import uncompress_image, undistort_image, concat_imgs, show_img
#from Vision.delete_images import del_pkl

clicked_points = []
directory = 'Vision/'
# Fonction de rappel pour gérer les clics de la souris
def mouse_callback(event, x, y, flags, param):
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:  # Clic gauche
        clicked_points.append([x, y])  # Ajouter les coordonnées du clic
        print(f"\nPoint selected : {(x, y)}")
        if len(clicked_points) == 4:
            print("\n4 coins sélectionnés !")

# Fonction Calibration
# INPUT: fichier pour enregistrer, chemin de l'image ref, taille image out
def calibrate_corners(calibration_file, reference_image, output_size):
    global clicked_points
    clicked_points = []

    # Si le fichier de calibration existe deja
    print(directory + calibration_file)
    if os.path.exists(directory + calibration_file):
        with open(directory + calibration_file, 'rb') as file:
            input_points = pickle.load(file) #Load input_points depuis le fichier
        #print("Calibration chargée depuis le fichier.")
    
    # Si il n'y a pas de calibration faite
    else:
        print("pkl file not found")
        # Charger image ref et convertir en niveau de gris
        #cv2.imshow("reference_image", reference_image)
        if reference_image is None:
            print("Pas d'image")
        gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

        # Créer lrea fenêtre et configurer le rappel de la souris
        cv2.namedWindow('Calibration')
        cv2.setMouseCallback('Calibration', mouse_callback)

        print("\nVeuillez cliquer sur les 4 coins de l'échiquier dans l'ordre :")
        print("1. Haut-gauche | 2. Haut-droit | 3. Bas-droit | 4. Bas-gauche")

        # Afficher l'image et attendre les 4 clics
        while True:
            cv2.imshow('Calibration', gray)

            # Quand 4 points sont cliqués, fermer la fenêtre
            if len(clicked_points) == 4:
                input_points = np.array(clicked_points, dtype=np.float32)
                cv2.destroyAllWindows()
                break

            # Toucher "Échap" pour quitter
            if cv2.waitKey(1) & 0xFF == 27:
                print("Calibration annulée.")
                cv2.destroyAllWindows()
                return None

        # Sauvegarder les points de calibration
        with open(directory + calibration_file, 'wb') as file:
            pickle.dump(input_points, file)
        print("Calibration done .")

    return input_points

# Fonction transformation projective 
def compute_transformation(input_points, output_size):
    #Output points de la taille de sortie 
    output_points = np.array([
        [0, 0],
        [output_size[1] - 1, 0],
        [output_size[1] - 1, output_size[0] - 1],
        [0, output_size[0] - 1]
    ], dtype=np.float32)
    # Calcul de la transformation perspective entre input et output points
    tform = cv2.getPerspectiveTransform(input_points, output_points)
    return tform

# Transformation projective (redressement)
def rectify_image(image, tform, output_size):
    return cv2.warpPerspective(image, tform, (output_size[1], output_size[0]))

# Chemin vers le répertoire "Downloads"
ImageDirectory ='Images'

# Image directory
def take_picture(robot, img_name):
    mtx, dist = robot.niryo.get_camera_intrinsics()
    img_compressed = robot.niryo.get_img_compressed()
    img_raw = uncompress_image(img_compressed)
    img_undistort = undistort_image(img_raw, mtx, dist)
    # Sauvegarder l'image
    
    output_path = os.path.join(ImageDirectory, str(img_name) + '.png')
    cv2.imwrite(output_path, img_undistort)
    print(output_path)
    robot.niryo.play_sound("learning_trajectory.wav")
    
    return img_undistort

def main():
    del_pkl(directory)
    robot = Robot()
    robot.move_to_obs_pose()
    take_picture(robot, "calibration_img")
    calibrate_corners("chessboard_calibration.pkl", cv2.imread("Images/calibration_img.png"), (800, 800))
    robot.move_to_V_pose()
    take_picture(robot, "V_calibration_img")
    calibrate_corners("V_calibration.pkl", cv2.imread("Images/V_calibration_img.png"), (400, 500))
    robot.move_to_v_pose()
    take_picture(robot, "v_calibration_img")
    calibrate_corners("v_calibration.pkl", cv2.imread("Images/v_calibration_img.png"), (400, 500))
    

    