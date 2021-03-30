# -*- coding: utf-8 -*-

# ------------------------------------------------------------------- #
# ---------------           Importations              --------------- #
# ------------------------------------------------------------------- #

from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import numpy as np
import random as rd
from math import sin, cos,pi
import json
import torch

# ------------------------------------------------------------------- #
# ---------------            fonctions                --------------- #
# ------------------------------------------------------------------- #

# BRUIT
def bruit_image(width,height,bruit, draw, couleur_bruit):
    """Fonction appliquant un bruit à une image PIL"""
    nombre_pixel_bruit = int(width*height*bruit)
    for pix in range(nombre_pixel_bruit):
        x = (rd.randint(0,width-1),rd.randint(0,height-1))
        y = (x[0]+2, x[1]+2)
        draw.rectangle((x,y), fill=couleur_bruit, width=1)

# SPIRULINES   
def sinusoide(x,y,longueur,angle,nb_points,amplitude,etirement):
    """Fonction définissant une sinusoide à dessiner"""
    pas = longueur/nb_points
    droite=[[x+k*cos(angle)*pas,y+k*sin(angle)*pas] for k in range(nb_points+1)]
    spiru=[]
    for k in range(nb_points+1):
        spiru.append((-sin(angle)*sin(pas*k*etirement)*amplitude+droite[k][0],cos(angle)*sin(pas*k*etirement)*amplitude+droite[k][1]))
    return spiru
    
def draw_sinusoide(x,y,longueur,angle,nb_points,amplitude,etirement, draw,
                   spiruline_color,epaisseur_filament):
    """Fonction dessinant une sinusoide sur une image PIL"""
    spiru = sinusoide(x,y,longueur,angle,nb_points,amplitude,etirement)
    draw.line(spiru,fill = spiruline_color,width=epaisseur_filament)


def image_spiruline(width, height, epaisseur_filament, bruit, background_color,spiruline_color, 
                    couleur_bruit,longueur_min,longueur_max,amplitude_min,amplitude_max,
                    etirement_min,etirement_max,filtre_gaussien,luminosite_min,luminosite_max,
                    nb_spiruline):
    """Fonction renvoyant une image PIL générée avec les paramètres"""
    # GENERATION DE L'IMAGE
    image = Image.new('RGBA', (width, height), background_color)
    draw = ImageDraw.Draw(image)
    bruit_image(width,height,bruit, draw, couleur_bruit)
    for cellule in range (nb_spiruline):
        longueur = int(rd.uniform(longueur_min,longueur_max))
        angle = rd.uniform(0,2*pi)
        amplitude = rd.uniform(amplitude_min, amplitude_max)
        etirement = rd.uniform(etirement_min,etirement_max)   
        x=rd.randint(0,width-1)
        y=rd.randint(0,height-1)
        draw_sinusoide(x,y,longueur,angle,100,amplitude,etirement, draw,
                       spiruline_color,epaisseur_filament)
    # FILTRES
    #Filtre Gaussien
    filtre = ImageFilter.GaussianBlur(radius=filtre_gaussien)
    image = image.filter(filtre)
    
    #Filtre Luminosité
    enhancer = ImageEnhance.Brightness(image)
    factor = rd.uniform(luminosite_min,luminosite_max)
    image = enhancer.enhance(factor)
    
    return image

def generateur_image(size, batch_size):
    """Générateur de batch d'images.
    """
    # DONNEES & PARAMETRES
    with open('parametres_image_JSON.json') as f:
        data = json.load(f)
    width = data['width']
    height = data['length']
    epaisseur_filament = data['epaisseur']
    bruit = data['bruit']
    nb_spiruline_max = data['nombre_filament_max']
    background_color = (int(data['background_color'][0]),int(data['background_color'][1]),int(data['background_color'][2]),int(data['background_color'][3]))
    spiruline_color = (int(data['spiruline_color'][0]),int(data['spiruline_color'][1]),int(data['spiruline_color'][2]),int(data['spiruline_color'][3]))
    couleur_bruit = (int(data['couleur_bruit'][0]),int(data['couleur_bruit'][1]),int(data['couleur_bruit'][2]),int(data['couleur_bruit'][3]))
    longueur_min = data["longueur_min"]
    longueur_max = data["longueur_max"]
    amplitude_min = data["amplitude_min"]
    amplitude_max = data["amplitude_max"]
    etirement_min = data["etirement_min"]
    etirement_max = data["etirement_max"]
    filtre_gaussien = data["filtre_gaussien"]
    luminosite_min = data["luminosite_min"]
    luminosite_max = data["luminosite_max"]
    
    x = []
    y = []
    for i in range(size):
        nb_spiruline = rd.randint(1,nb_spiruline_max)
        classe = int(nb_spiruline//10)
        image = image_spiruline(width, height, epaisseur_filament, bruit, background_color,spiruline_color, 
                    couleur_bruit,longueur_min,longueur_max,amplitude_min,amplitude_max,
                    etirement_min,etirement_max,filtre_gaussien,luminosite_min,luminosite_max,
                    nb_spiruline)
            
        img_array = np.asarray(image) # Récupération de la matrice numpy de image
        # Changement des axes pour correspondre au DataSetLoader de Pytorch en terme de dimensions
        red_array = img_array[:, :, 0]
        green_array = img_array[:, :, 1]
        blue_array = img_array[:, :, 2]
        image = np.asarray([red_array, green_array, blue_array])
        
        x.append(image)
        y.append(classe)
        if len(x) == batch_size or i == size-1:
            # Normalisation sur le batch
            l_pixel = np.asarray([im for im in x]) 
            x_min, x_max = l_pixel.min(), l_pixel.max()
            x = (l_pixel - x_min)/(x_max - x_min)
            # Conversion du batch et de la liste des labels en Tensor Pytorch
            x = torch.Tensor(x)
            y = torch.tensor(y, dtype=torch.long) # Important de préciser le type
            yield x, y
            x = []
            y = []
    
    