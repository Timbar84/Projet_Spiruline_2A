# -*- coding: utf-8 -*-
import json

dict_reseau = {
    "nb_epoch":15,
    "learning_rate":0.01,
    "step_size":7,
    "gamma":0.1,
    "train_size":8,
    "val_size":1,
    "nombre_classe":5,
    "batch_size":4,
    "pretrained":1
    }

dict_image = {
    "width":224,
    "length":224,
    "epaisseur":2,
    "bruit":0.1,
    "nombre_filament_max":40,
    "background_color":["134","180","152","255"],
    "spiruline_color":["107","161","125","255"],
    "couleur_bruit":["131","177","149","255"],
    "longueur_min":45,
    "longueur_max":55,
    "amplitude_min":1.9,
    "amplitude_max":2.1,
    "etirement_min":0.43,
    "etirement_max":0.58,
    "filtre_gaussien":0.8,
    "luminosite_min" : 0.8,
    "luminosite_max" : 1.2
    }

out_file = open("parametres_image_JSON.json", "w")
json.dump(dict_image, out_file, indent = 6)
out_file.close()
out_file = open("parametres_ResNet18.json", "w")
json.dump(dict_reseau, out_file, indent = 6)
out_file.close()