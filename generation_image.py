# -*- coding: utf-8 -*-

from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import random as rd
from math import sin, cos,pi

alphabet = list("abcdefghijklmnopqrstuvwxyz")

def generer_xy(width, height):
    lg_min = 50
    x = (rd.randint(0,width),rd.randint(0,height))
    x_ = min(x[0]+lg_min, width), min(x[1]+lg_min, height) # Pour donner une longueur minimum
    y = (rd.randint(x_[0],width),rd.randint(x_[1],height))
    return x,y

def generer_nom_fich():
    nom = ""
    for i in range(rd.randint(10, 40)):
        U = rd.uniform(0,1)
        if U<0.7:
            nom += str(rd.randint(0,9))
        else:
            nom += rd.sample(alphabet,1)[0]
    return nom
    
## Paramètres
width = 224
height = 224
epaisseur_filament = 3
bruit = 0.1
nombre_filament_max = 36

## Couleurs
# Pour le fond :
background_color = (134, 180, 152, 255)
# Pour la spiruline :
spiruline_color = (107, 161, 125, 255)
couleur_bruit = (131,177,149, 255)

def generer_image(nombre_spiruline, categorie, nom_fich):
    """Genere une image de spiruline avec [nombre_spiruline] filament sur l'image,
    il faut ensuite renseigner deux arguments pour enregistrer l'image. A savoir
    categorie qui prend ['train' ou 'val'] pour le dossier et un nom de fichier 
    [nom_fich]"""
    ## Creation image
    image = Image.new('RGBA', (width, height), background_color)
    ## Obtention du contexte graphique
    draw = ImageDraw.Draw(image)
    
    # Fond plus foncé
    nombre_pixel_bruit = int(width*height*bruit)
    for pix in range(nombre_pixel_bruit): # bruit
        # image.putpixel((rd.randint(0,width-1),rd.randint(0,height-1)), fond_fonce)
        x = (rd.randint(0,width-1),rd.randint(0,height-1))
        y = (x[0]+2, x[1]+2)
        draw.rectangle((x,y), fill=couleur_bruit, width=1)
        
    # ## Tracer des filaments
    # nb_filament = nombre_spiruline
    # for filament in range(nb_filament):
    #     U = rd.uniform(0,1)
    #     xy = generer_xy(width, height)
    #     if U>0.7:
    #         draw.line(xy, fill=spiruline_color, width=epaisseur_filament)
    #     else:
    #         ang_deb = rd.randint(0, 180)
    #         ang_fin = ang_deb + rd.randint(60,100)
    #         draw.arc(xy, ang_deb, ang_fin, fill=spiruline_color, width=epaisseur_filament)
    
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
    
    for cellule in range (nombre_spiruline):
        longueur = int(rd.uniform(45,55))
        angle = rd.uniform(0,2*pi)
        amplitude = rd.uniform(1.9,2.1)
        etirement = rd.uniform(0.43,0.58)   
        x=rd.randint(0,width-1)
        y=rd.randint(0,height-1)
    
        draw_sinusoide(x,y,longueur,angle,100,amplitude,etirement)
    
    ## Filtre flou
    filtre = ImageFilter.GaussianBlur(radius=0.9)
    image = image.filter(filtre)
    
    ## Modification Couleur, Luminosité, Saturation
    
    def create_hls_array(image):
        """
        Cette fonction va convertir une image RGB en une nouvelle HLS, sous le format
        d'un tableau Numpy. 
        HLS = Couleur, Lumière, Saturation.
        """
     
        
        pixels = image.load()
    
        hls_array = np.empty(shape=(image.height, image.width, 3), dtype=float)
    
        for row in range(0, image.height):
    
            for column in range(0, image.width):
    
                rgb = pixels[column, row]
    
                hls = colorsys.rgb_to_hls(rgb[0]/255, rgb[1]/255, rgb[2]/255)
    
                hls_array[row, column, 0] = hls[0]
                hls_array[row, column, 1] = hls[1]
                hls_array[row, column, 2] = hls[2]
    
        return hls_array

    def image_from_hls_array(hls_array):
        """
        Cette fonction va appliquer modifier aléatoirement les 3 paramètres H,L,S.
        Ensuite, elle va générer une nouvelle image RGB format PIL.
        """
        
        # COULEUR # :   cercle chromatique des couleurs. Un tour complet correpond à l'unité
        modifCouleur = rd.uniform(-0.1,0.1)
        # LUMIERE # :   une valeur positive augmente la luminosité, une valeur négative la diminue.  0=rien ne change / 1=tout blanc /-1=tout noir. 
        modifLumiere = rd.uniform(-0.2,0.3)
        # SATURATION # :une valeur positive augmente la saturation, une valeur négative la diminue (vers la couleur complementaire).  0=rien ne change / 1=saturation totale / -1=saturation totale negative.
        modifSaturation =rd.uniform(-0.15,0.2)
 
        new_image = Image.new("RGB", (hls_array.shape[1], hls_array.shape[0]))
    
        for row in range(0, new_image.height):
            for column in range(0, new_image.width):
    
                rgb = colorsys.hls_to_rgb(hls_array[row, column, 0]+modifCouleur,
                                          hls_array[row, column, 1]+modifLumiere,
                                          hls_array[row, column, 2]+modifSaturation)
    
                rgb = (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
                new_image.putpixel((column, row), rgb)
        return new_image

    image_hls = create_hls_array(image)
    image = image_from_hls_array(image_hls)
    
    ## Convertissement et sauvegarde en jpg
    image = image.convert("RGB")
    classe = nombre_spiruline
    path = "./data/petit_18_classes/{}/{}/{}.jpg".format(categorie, classe, nom_fich)
    image.save(path)

proportion_train = 0.70
nb_image_total_par_classe = 20
for classe in range(1,19):
    for i in range(int(nb_image_total_par_classe*proportion_train)):
        categorie = 'train'
        generer_image(classe, categorie, generer_nom_fich())
    for i in range(int(nb_image_total_par_classe*(1-proportion_train))):
        categorie ='val'
        generer_image(classe, categorie, generer_nom_fich())
        
    print(classe, " done")
# image.show()
