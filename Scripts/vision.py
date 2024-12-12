from pyniryo import NiryoRobot, PoseObject
from pyniryo.vision import uncompress_image, undistort_image, concat_imgs, show_img
import cv2
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("name")
args = parser.parse_args()

# Chemin vers le répertoire "Downloads"
directory ='.'

# Changer le répertoire courant




# Image directory

os.chdir('.')

robot = NiryoRobot('192.168.94.1')
mtx, dist = robot.get_camera_intrinsics()
    # Getting image
img_compressed = robot.get_img_compressed()
# Uncompressing image
img_raw = uncompress_image(img_compressed)
    # Undistorting
img_undistort = undistort_image(img_raw, mtx, dist)

os.chdir(directory)

# Sauvegarder l'image
output_path = os.path.join('.', args.name + '.png')
cv2.imwrite(output_path, img_undistort)