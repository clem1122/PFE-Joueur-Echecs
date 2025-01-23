import os
import glob

def del_im(path):
    files_to_delete = glob.glob(os.path.join(path, "[0-9]*.png"))

    # Supprimer les fichiers
    for file in files_to_delete:
        os.remove(file)
        print(f"Fichier supprimé : {file}")

def del_pkl(path):
    files_to_delete = glob.glob(os.path.join(path, "*.pkl"))

    # Supprimer les fichiers
    for file in files_to_delete:
        os.remove(file)
        print(f"Fichier supprimé : {file}")