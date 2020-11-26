# -*- coding: utf-8 -*-

from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import random as rd

def generer_xy(width, height):
    x = (rd.randint(0,width),rd.randint(0,height))
    y = (rd.randint(x[0],width),rd.randint(x[1],height))
    return x,y
    
## Paramètres
width = 224
height = 224
epaisseur_filament = 3
bruit = 0.1
nombre_filament_max = 36
# nombre_tache_noire = 2
# grosseur_tache = 5

## Couleurs
# Pour le fond :
background_color = (134, 180, 152, 255)
# Pour la spiruline :
spiruline_color = (107, 161, 125, 255)
couleur_bruit = (131,177,149, 255)
# fond_tache = (134,181,147, 255)
# a= -5
# fond_tache2 = (134+a,181+a,147+a, 255)

## Creation image
image = Image.new('RGBA', (width, height), background_color)
## Obtention du contexte graphique
draw = ImageDraw.Draw(image)

## Bruit
# Taches
# for tache in range(nombre_tache_noire):
#     ell_a = (rd.randint(0,width),rd.randint(0,height))
#     ell_b = (ell_a[0]+grosseur_tache,ell_a[1]+grosseur_tache)
#     draw.ellipse((ell_a,ell_b), fill = fond_tache)
#     ell_a_2 = (ell_a[0]+int(grosseur_tache/2),ell_a[1]+int(grosseur_tache/2))
#     ell_b_2 = (ell_a[0]-int(grosseur_tache/2),ell_b[1]-int(grosseur_tache/2))
#     fond_tache = (134+a,181+a,147+a, 255)
#     draw.ellipse((ell_a,ell_b), fill = fond_tache2)
    
# Fond plus foncé
nombre_pixel_bruit = int(width*height*bruit)
for pix in range(nombre_pixel_bruit):
    # image.putpixel((rd.randint(0,width-1),rd.randint(0,height-1)), fond_fonce)
    x = (rd.randint(0,width-1),rd.randint(0,height-1))
    y = (x[0]+2, x[1]+2)
    draw.rectangle((x,y), fill=couleur_bruit, width=1)
    
## Tracer des filaments
nb_filament = rd.randint(1, nombre_filament_max)
for filament in range(nb_filament):
    U = rd.uniform(0,1)
    xy = generer_xy(width, height)
    if U>0.7:
        draw.line(xy, fill=spiruline_color, width=epaisseur_filament)
    else:
        ang_deb = rd.randint(0, 180)
        ang_fin = rd.randint(ang_deb, 180)
        
        draw.arc(xy, ang_deb,ang_fin, fill=spiruline_color, width=epaisseur_filament)

## Filtre flou
filtre = ImageFilter.GaussianBlur(radius=0.9)
image = image.filter(filtre)

image.show()
# image.save("cercleTrigo.png", "png")










