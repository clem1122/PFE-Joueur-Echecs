from pyniryo import NiryoRobot, PoseObject
from pyniryo.vision import uncompress_image, undistort_image, concat_imgs, show_img
import cv2
import os
import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("name")
# args = parser.parse_args()

# Chemin vers le répertoire "Downloads"
directory ='Image'
os.chdir(directory)
# Changer le répertoire courant

# Image directory
def take_picture(robot, img_number):
    
    mtx, dist = robot.niryo.get_camera_intrinsics()
    img_compressed = robot.niryo.get_img_compressed()
    img_raw = uncompress_image(img_compressed)
    img_undistort = undistort_image(img_raw, mtx, dist)
    # Sauvegarder l'image
    output_path = os.path.join('.', str(img_number) + '.png')
    cv2.imwrite(output_path, img_undistort)
    return img_undistort
