from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import numpy as np
import random as rd
from math import sin, cos,pi

# DONNEES & PARAMETRES
import json
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

# GENERATION DE L'IMAGE
image = Image.new('RGBA', (width, height), background_color)
draw = ImageDraw.Draw(image)

# BRUIT
def bruit_image(width,height,bruit):
    nombre_pixel_bruit = int(width*height*bruit)
    for pix in range(nombre_pixel_bruit):
        x = (rd.randint(0,width-1),rd.randint(0,height-1))
        y = (x[0]+2, x[1]+2)
        draw.rectangle((x,y), fill=couleur_bruit, width=1)

bruit_image(width,height,bruit)

# SPIRULINES   
def sinusoide(x,y,longueur,angle,nb_points,amplitude,etirement):
    pas = longueur/nb_points
    droite=[[x+k*cos(angle)*pas,y+k*sin(angle)*pas] for k in range(nb_points+1)]
    spiru=[]
    for k in range(nb_points+1):
        spiru.append((-sin(angle)*sin(pas*k*etirement)*amplitude+droite[k][0],cos(angle)*sin(pas*k*etirement)*amplitude+droite[k][1]))
    return spiru
    
def draw_sinusoide(x,y,longueur,angle,nb_points,amplitude,etirement):
    spiru = sinusoide(x,y,longueur,angle,nb_points,amplitude,etirement)
    draw.line(spiru,fill = spiruline_color,width=epaisseur_filament)

nb_spiruline = rd.randint(1,nb_spiruline_max)

for cellule in range (nb_spiruline):
    longueur = int(rd.uniform(45,55))
    angle = rd.uniform(0,2*pi)
    amplitude = rd.uniform(1.9,2.1)
    etirement = rd.uniform(0.43,0.58)   
    x=rd.randint(0,width-1)
    y=rd.randint(0,height-1)

    draw_sinusoide(x,y,longueur,angle,100,amplitude,etirement)
    
# FILTRES
    
#Filtre Gaussien
filtre = ImageFilter.GaussianBlur(radius=filtre_gaussien)
image = image.filter(filtre)

#Filtre Luminosité
enhancer = ImageEnhance.Brightness(image)
factor = rd.uniform(luminosite_min,luminosite_max)
image = enhancer.enhance(factor)

image.show()
# image.save("cercleTrigo.png", "png")