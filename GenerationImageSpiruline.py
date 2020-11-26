# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 14:41:13 2020

@author: Antoine
"""

from PIL import*

#new_im = Image.new('RGB',(1280,720),(134,180,152))
#new_im.show()

#image1 = Image.open("9340.png")
#image1.show()

"""Graphisme 2D avec Pillow."""

from PIL import Image, ImageDraw
from random import *
from math import sqrt

"""
===============================================================================
Données / Réglages
===============================================================================
"""

largeur = 1280
hauteur = 720
FondImage = (255, 255, 255,0)
CouleurSpiruline = (107,161,125,0)
#CouleurSpiruline = (0,0,0,0)
Background=(134,180,152)
image = Image.new('RGBA', (largeur, hauteur), Background)

"""
===============================================================================
Génération spiruline 
===============================================================================
"""
#draw = ImageDraw.Draw(image)

for i in range(1000):
    image = Image.new('RGBA', (largeur, hauteur), Background)
    draw = ImageDraw.Draw(image)
    longueur=0
    s=randint(1, 25)
    for k in range(s):
        xA = uniform(0,largeur)
        yA = uniform(0,hauteur)
       
        xB = uniform(xA-largeur//2,xA+largeur//2)
        yB = uniform(yA-largeur//2,yA+largeur//2)
        
        draw.line((xA,yA,xB,yB),fill=CouleurSpiruline,width=3)
        longueur+=sqrt((xB-xA)**2+(yB-yA)**2)
    
    longueur = round(longueur,2)
    longueur=str(longueur)
    #nom ="nb spiru_"+str(s)+'numero_'+ str(i)
    nom =str(i)+"_"+str(longueur)
    image.save(nom+".png")
    #image.save(nom + "_longueur_totale=" + longueur+".png")
    image.show()
    
image.show()
#image.save("cercleTrigo.png", "png")
