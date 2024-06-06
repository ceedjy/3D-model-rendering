# coding: utf-8
# code ecrit par Cassiopée Gossin - L1 CMI informatique (2023) et Raphael Tournafond - L1 CMI informatique (2017)

#Importations necessaires
import numpy as np
from numpy.linalg import inv
from math import cos, sin, tan, pi, floor, inf, sqrt
#from PIL import Image #Importation bibliothèque image (TP1-Math202)
import copy

#on import une image pour la texture 
import matplotlib
from matplotlib import image

#on import un autre fichier pour les ombres
from raster_ombre2 import *

# Partie de Raphael 

# 1. Projection

# Fonctions transformation dans le monde

def dilatation(v, Cx, Cy, Cz):
    """
    Cx, Cy, Cz les coefficients de dilatation par axe
    v : les coordonnées x,y,z du point
    """
    #Multiplication avec numpy (.dot(a,b))
    return np.dot(v, [[Cx,0,0],[0,Cy,0],[0,0,Cz]])

def rotation(v, axe, angle):
    """
    v = [x,y,z] tab
    axe = "x", "y" ou "z" str
    angle = float en degré
    """

    a = (angle*pi)/180 #Passage de l'angle en radian
    c = cos(a)
    s = sin(a)

    #Test pour trouver la bonne matrice de rotation
    if axe == "x":
        m_rotation = [[1,0,0],[0,c,-s],[0,s,c]]
    elif axe == "y":
        m_rotation = [[c,0,s],[0,1,0],[-s,0,c]]
    elif axe == "z":
        m_rotation = [[c,-s,0],[s,c,0],[0,0,1]]
    else:
        print("Erreur axe: 'x', 'y' ou 'z'")
        m_rotation = [[1,0,0],[0,1,0],[0,0,1]]
        
    return np.dot(v, m_rotation)

def translation(v, Tx, Ty, Tz):
    """
    v = [x,y,z,1]
    Tx, Ty, Tz = entier
    """

    #Si la coordonnée est toujours de taille 1*3 et non 1*4
    if len(v) != 4:
        return print("Erreur: v = [x,y,z,1]")
    return np.dot(v, [[1,0,0,0],[0,1,0,0],[0,0,1,0],[Tx,Ty,Tz,1]])

def matrice_T(d, r, t):
    """
    d = [dx,dy,dz] Les coefficients de dilatation sur chaque axe
    r = [ax,ay,az] Les angles de rotation autour de chaque axe
    t = [tx,ty,tz] Les coefficients de translation sur chaque axe
    """
    
    tra = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[t[0],t[1],t[2],1]] #Matrice translation
    dil = [[d[0],0,0,0],[0,d[1],0,0],[0,0,d[2],0],[0,0,0,1]] #Matrice dilatation
    a = (r[0]*pi)/180
    c = cos(a)
    s = sin(a)
    Rx = [[1,0,0,0],[0,c,-s,0],[0,s,c,0],[0,0,0,1]]
    a = (r[1]*pi)/180
    c = cos(a)
    s = sin(a)
    Ry = [[c,0,s,0],[0,1,0,0],[-s,0,c,0],[0,0,0,1]]
    a = (r[2]*pi)/180
    c = cos(a)
    s = sin(a)
    Rz = [[c,-s,0,0],[s,c,0,0],[0,0,1,0],[0,0,0,1]]
    rot = np.dot(Rx,Ry)
    rot = np.dot(rot, Rz) #Matrice rotation

    #Multiplication des trois sous-matrices
    T = np.dot(dil, rot)
    T = np.dot(T, tra)
    return T

#Fonction local vers monde n'utilisant pas la fonction Matrice_T afin
#d'illustrer le fonctionnement

def local_vers_monde_v1(v, d, r, t):
    """
    v = [x,y,z] Les coordonnées de base du point
    d = [dx,dy,dz] Les coefficients de dilatation sur chaque axe
    r = [ax,ay,az] Les angles de rotation autour de chaque axe
    t = [tx,ty,tz] Les coefficients de translation sur chaque axe
    """
    
    v_proj = dilatation(v,d[0],d[1],d[2])
    tab = ["x","y","z"]
    for i in range(3):
            v_proj = rotation(v_proj, tab[i], r[i])
    v_proj = np.insert(v_proj,v_proj.size,1)
    point_monde = translation(v_proj, t[0], t[1], t[2])
    return point_monde


# Procedure détaillée de la projection

#Fonction local vers monde utilisant fonction Matrice_T
#plus pertinante

def local_vers_monde(point, d, r, t):
    """
    point = [x,y,z,1] Les coordonnées de base du point
    d = [dx,dy,dz] Les coefficients de dilatation sur chaque axe
    r = [ax,ay,az] Les angles de rotation autour de chaque axe
    t = [tx,ty,tz] Les coefficients de translation sur chaque axe
    """
    #Deduction de la matrice transformation
    T = matrice_T(d, r, t)
    #multiplication
    point_monde = np.dot(point,T)
    
    return point_monde

def monde_vers_camera(point,dc,tc,rc):
    """
    Projection du point dans le monde vers la camera
    """

    #Projection camera dans monde puis monde dans camera
    camera_monde = matrice_T([1,1,1], rc, tc) #V
    camera_local = inv(camera_monde) #V^-1

    #Projection du point dans l'espace camera
    point_camera = np.dot(point, camera_local)
    
    return point_camera

def matrice_P(fov, n, f):
    """
    Calcul de la matrice perspective tel que vu dans le wiki
    fov : champ de vision en degré
    n : distance premier plan
    f : distance arriere plan
    """
    s = 1./(tan((fov/2.)*(pi/180.)))
    return [[s,0,0,0],[0,s,0,0],[0,0,-(f+n)/(f-n),-2*(f*n)/(f-n)],[0,0,-1,0]]

def camera_vers_ecran(point,fov,proche,lointain):
    """
    Projection du point sur l'ecran de la camera
    """
    
    #Creation de la matrice perspective
    perspective = matrice_P(fov, proche, lointain)

    #Projection du point sur l'ecran
    point_ecran = np.dot(perspective, point)
    w = point_ecran[3]
    if(abs(w) > 1e-16):
        return (point_ecran[:-1] / w, w) #[:-1] car l'on peut desormait travaiiler sur (x,y,z)
    return (point_ecran[:-1], 1.) #[:-1] car l'on peut desormait travaiiler sur (x,y,z)
                            # et non (x,y,z,1)

def ecran_vers_normalise(point, largeur, hauteur):
    """
    Transformation du point aux coordonnees normalisees
    """
    
    p = point
    l = largeur
    h = hauteur
    #Si le point est comprit dans l'image
    if (abs(p[0]) <= l/2) or (abs(p[1]) <= h/2):
        #Transformation normalisée (NDC space)
        p[0] = (p[0]+l/2)/l
        p[1] = (p[1]+h/2)/h
        return p
    return None

def normalise_vers_grille(point, l, h):
    """
    Transformation du point aux coordonnees pixel
    """
    
    p = point
    p[0] = floor(p[0]*l)
    p[1] = floor((1-p[1])*h)
    return p

# Resultat final

def procedure_local_grille(point_local,index,couleur,dp,rp,tp,dc,rc,tc,fov,proche,lointain,largeur,hauteur):
    """
    Transformation totale du point local vers l'espace raster
    """
    
    point_monde = local_vers_monde(point_local,dp,rp,tp)
    point_camera = monde_vers_camera(point_monde,dc,rc,tc)
    (point_ecran, w) = camera_vers_ecran(point_camera,fov,proche,lointain)
    point_normalise = point_ecran
    point_normalise[0] = (point_normalise[0] + 1.) / 2.
    point_normalise[1] = (point_normalise[1] + 1.) / 2.
    #point_normalise = ecran_vers_normalise(point_ecran, largeur, hauteur)
    point_grille = normalise_vers_grille(point_normalise, largeur, hauteur)
    x = point_grille[0]
    y = point_grille[1]
    coords = (int(x),int(y),point_grille[2])
    return ((coords,index,couleur), w)   #Ajout d'un index et de la couleur necessaire
                                    #à la visualisation
    
# 2. Visualisation

# Recupération des coordonnées comprisent dans un triangle

def ordre(s1,s2,s3):
    """
    Retourne l'ordre des sommets grace à leur index
    s = ((x,y,z),i)
    """
    
    tuples = [s1,s2,s3]
    return sorted(tuples, key=lambda tuples: tuples[1])

def boite(s1,s2,s3,width,height):
    """
    Génération de la plus petite boite englobant le triangle
    """
    
    boite_start = (min(s1[0],s2[0],s3[0]),min(s1[1],s2[1],s3[1]))
    boite_end = (max(s1[0],s2[0],s3[0]),max(s1[1],s2[1],s3[1]))
    if boite_start[0] < 0:
        boite_start = (0, boite_start[1])
    if boite_start[1] < 0:
        boite_start = (boite_start[0], 0)
    if boite_end[0] >= width:
        boite_end = (width-1, boite_end[1])
    if boite_end[1] >= height:
        boite_end = (boite_end[0], height-1)
    
    return (boite_start,boite_end) #(xmin,ymin),(xmax,ymax)

def interieur_bordure(s1,s2,pixel):
    """
    Calcul le déterminant D pour ensuite tester si le pixel est dans le triangle
    """
    
    return ((pixel[0] - s1[0]) * (s2[1] - s1[1]) - (pixel[1] - s1[1]) * (s2[0] - s1[0]))

def liste_pixels_coeffs(s1,s2,s3,width,height,w1,w2,w3):
    """
    Retourne la listes des pixels dans le triangle ainsi que leurs coefficients
    par rapport à chaque sommets
    """
    
    b = boite(s1,s2,s3,width,height)
    points = []
    coeffs = []
    #Aire du triangle complet
    aire = interieur_bordure(s1,s2,s3)
    #Parcours de la bounding box
    #filtre les triangles ne faisant pas face
    #on pourrait intervertir à la place pour faire les deux faces
    if aire > 0:
        for i in range(b[0][0],b[1][0]+1):
            for j in range(b[0][1],b[1][1]+1):
                #Calcul des sous-aires
                aire3 = interieur_bordure(s1,s2,(i,j))
                aire1 = interieur_bordure(s2,s3,(i,j))
                aire2 = interieur_bordure(s3,s1,(i,j))
                #Si le pixel est à l'interieur des trois bords du triangle
                if aire1 >= 0 and aire2 >= 0 and aire3 >= 0:
                    #Coordonnées barycentriques
                    tot = aire1 / w1 + aire2 / w2 + aire3 / w3
                    coeff = ((aire1/ w1 /tot,aire2/w2/tot,aire3/w3/tot))
                    #Calcul de la profondeur par rapport aux coords barycentriques
                    z = coeff[0]*s1[2]+coeff[1]*s2[2]+coeff[2]*s3[2]
                    points.append((i,j,z))
                    coeffs.append(coeff)
    return (points,coeffs)

def calculer_rgb(c1,c2,c3,liste,normale, triangle, coor_monde):  
    #ecrit en grande partie par Cassiopee Gossin
    """
    Calcul de la couleur du pixel
    entree : c1/2/3 : les couleurs des differents sommets du triangle (R,G,B)
            liste : liste de (point,coeff), resulatat de la fonction precedente 
            triangle : la liste des triangle tel que decrite dans la variable globale 
            coor_monde : la position des 3 points qui consituent le triangle 
                dans le monde apres les transformations, un tableau
    """
    
    #Liste des points du triangle et de leur coeff
    finale = liste[0]
    k = 0
    for i in range (0, len(liste[1])):
        cf1 = liste[1][i][0] 
        cf2 = liste[1][i][1]
        cf3 = liste[1][i][2]
        
        #avec les coordonnees barycentriques, on recupere la position du point 
        #dans le monde apres les transformations
        pointA = coor_monde[0]
        pointB = coor_monde[1]
        pointC = coor_monde[2]
        point = retrouver_point(cf1, cf2, cf3, pointA, pointB, pointC)
        
        couleur = ()
        
        #dans le cas ou il y a une sphere lisse
        if ask_lisse_sphere:
            normaleA = creer_vecteur([centre_transfo, pointA])
            normaleB = creer_vecteur([centre_transfo, pointB])
            normaleC = creer_vecteur([centre_transfo, pointC])
            normale_s = normale_point_sphere(cf1, cf2, cf3, normaleA, normaleB, normaleC)
            #rotation de la normale pour correspondre aux coordonnees du monde 
            true_normale = rotation(normale_s, "x", triangle[2][0])
            true_normale = rotation(true_normale, "y", triangle[2][1])
            true_normale = rotation(true_normale, "z", triangle[2][2])
        else: 
            #rotation de la normale pour correspondre aux coordonnees du monde 
            true_normale = rotation(normale, "x", triangle[2][0])
            true_normale = rotation(true_normale, "y", triangle[2][1])
            true_normale = rotation(true_normale, "z", triangle[2][2])
            
        vec_light = vecteur_light(point, coor_src_lum)
        
        #on applique la texture
        info_textu = applique_texture(cf1, cf2, cf3, texture)
        
        #pour les speculaires 
        vect_H = vecteur_H(vec_light, vecteur_vue(point, coor_camera))
        
        #Pour chaque composante (r, g et b)
        for j in range(3):
            #Nouvelle couleur
            #on differencie si il y a une texture ou non
            if ask_texture == False:
                c = floor(cf1*c1[j]+cf2*c2[j]+cf3*c3[j])
            else:
                c = (info_textu[j])
            #on cree les differents coefficients de lumiere
            lum_ambiante = 0.1 #la lumiere ambiante
            lum_diffuse = 1.0 * illumination(vec_light, true_normale) #la lumiere diffuse
            lum_speculaire = speculaire(vect_H, true_normale, alpha) #la lumiere speculaire
            #on differencie si il y a une ombre ou non 
            if not ask_ombre:
                val_rgb = floor(c * (lum_ambiante + lum_diffuse + lum_speculaire)) #on rajoute l'illumination avec une multiplication (lumiere ambiente + diffuse + specuaire)
            else:
                if est_visible(point, image_lumiere, cam_lum): #si le point est vu par la lumiere 
                    val_rgb = floor(c * (lum_ambiante + lum_diffuse + lum_speculaire)) #on rajoute l'illumination avec une multiplication (lumiere ambiente + diffuse + specuaire)
                else: #si le point est dans l'ombre
                    val_rgb = floor(c * lum_ambiante)
            #ayant rajoute la lumiere ambiante, on verifie si la couleur ne depasse pas 255
            if val_rgb > 255:
                val_rgb = 255
            couleur += (val_rgb,) 
        #Correspondance du point et de la couleur (ecrase les coord barycentriques)
        finale[k] = (finale[k],couleur) 
        k += 1
    return finale


def remplir(s1,s2,s3, normale, triangle, width, height, w1, w2, w3, coor_monde):
    """
    Renvoie la liste des pixels ainsi que leur couleur
    s = ((x,y,z),i,(r,g,b))
    """
    
    o = ordre(s1,s2,s3)
    liste = liste_pixels_coeffs(o[0][0],o[1][0],o[2][0],width,height,w1,w2,w3)
    return calculer_rgb(s1[2],s2[2],s3[2],liste, normale, triangle, coor_monde)

# Image

def creer_image(l,h,fond,fichier):
    """s
    Création d'une image vierge
    l = largeur
    h = hauteur
    fichier = fichier de sortie
    fond = couleur du fond (r,g,b)
    """

    #ouverture du fichier en mode lecture
    image = open(fichier,"w")
    image.write("P3\n"+str(l)+" "+str(h)+"\n255\n")
    #Mettre tous les pixels à la couleur du fond
    for i in range(h):
        for j in range(l):
            image.write(str(fond[0])+" "+str(fond[1])+" "+str(fond[2])+"\n")
    #fermeture du fichier
    image.close()
    return None

def rendu(liste,fichier):
    """
    Ecrit les nouveaux pixel sur l'image
    """
    
    im = Image.open(fichier)
    for i in liste:
        x = i[0][0]
        y = i[0][1]
        c = i[1]
        #Mise à jour des nouveaux pixels avec leur couleur
        im.setPixel(x,y,c)
        
    im.save(fichier)
    return None

def rendu_total(l,h,fond,fichier,liste):
    """
    Créé l'image en partant des dimensions et de la liste de pixels
    """
    
    creer_image(l,h,fond,fichier)
    rendu(liste,fichier)
    return None

# Implementation du z-buffer

def initialisation_z_buffer(l,h):
    """
    Creation d'un tableau 2D avec chaque case = -inf
    """
    z_buffer = []
    #Ligne
    for i in range(0,h):
        z_buffer.append([])
        for j in range(0,l):
            z_buffer[i].append(-inf)
    return z_buffer

def cache(z_buffer,p):
    """
    Test si un pixel est cache
    """
    
    if p[2] <= z_buffer[p[1]][p[0]]:
        return True
    return False

def modifier_z_buffer(z_buffer,p):
    """
    Modifie la case avec la nouvelle profondeur
    """
    if p[0] < len(z_buffer[p[1]]):
        z_buffer[p[1]][p[0]] = p[2]
    return z_buffer

def rendu_z_buffer(l,h,liste,fichier,z_buffer):
    """
    Rendu de l'image en prenant compte le z-buffer
    """

    #Ouverture de l'image
    #im = Image.open(fichier)
    with open(fichier,"r") as image:
        data = image.readlines()
    for i in liste:
        x = i[0][0]
        y = i[0][1]
        z = i[0][2]
        c = i[1]
        #tests z-buffer
        if not cache(z_buffer,(x,y,z)):
            z_buffer = modifier_z_buffer(z_buffer,(x,y,z))
            data[3 + x + y * l] = str(c[0]) + " " + str(c[1]) + " " + str(c[2]) + "\n"
            #im.setPixel(x,y,c)
    
    #Sauvegarde de l'image
    with open(fichier,"w") as image:
        image.writelines(data)
    return None

#-----------------------------------------------------------------------------

# Partie de Cassiopée 

# 1 : le vecteur normale du triangle

def calcul_normale(coorv1, coorv2):
    '''
    calcul la normale d'un triangle avec les coordonnées de 2 vecteurs 
    Entrees :  coorv1 : cordonnnees du vecteur 1
               corv2 : cordonnnees du vecteur 2
    sortie : (x, y, z) coordonnees qui représentent les coordonnees du vecteur 
    normal du triangle
    '''
    #récupération des coordonnées du vecteur 1
    a = coorv1[0]
    b = coorv1[1]
    c = coorv1[2]
    #récupération des coordonnées du vecteur 2
    d = coorv2[0]
    e = coorv2[1]
    f = coorv2[2]
    #application de la formule du produit vectoriel pour calculer la normale du triangle
    x = b*f-c*e
    y = c*d-a*f
    z = a*e-b*d
    #pour obtenir la normale, il faut diviser ce vecteur par sa norme
    #calcul de la norme de ce vecteur
    norme_vec = sqrt(x**2 + y**2 + z**2)
    #division du vecteur par sa norme
    x = x/norme_vec
    y = y/norme_vec
    z = z/norme_vec
    return (x, y, z)

def creer_vecteur(coor_points):
    '''
    cree un vecteur avec les coordonnees des points  
    Entrees : coor_points : les coordonnees des points A et B qui vont nous 
    permettre de créer un vecteur (exemple : [(x, y, z), (x', y', z')]), tableau de tuple
    sortie : (x, y, z) les coordonnees du vecteur qui part du point A et qui va
    juqu'au point B
    '''
    #récupération des coordonnées du point A
    a = coor_points[0][0]
    b = coor_points[0][1]
    c = coor_points[0][2]
    #récupération des coordonnées du point B
    d = coor_points[1][0]
    e = coor_points[1][1]
    f = coor_points[1][2]
    #calcul des coordonnées du vecteur AB
    x = d-a
    y = e-b
    z = f-c
    return (x, y, z) 

def normale_triangles(triangle):
    '''
    calcule la normale du triangle renseigne dans 'triangle'
    entrees : triangles : la variable globale
    sorties : normale : la normale du triangles renseignes dans 'triangles'
    '''
    #pour chaque triangles de 'triangles'
    for i in range (0, len(triangles)): #verifier si il faut garder cette ligne ou non 
        #on reprend les points decrits dans 'triangles'
        pointA = triangle[0][0][0]
        pointB = triangle[0][1][0]
        pointC = triangle[0][2][0]
        #on cree les vecteurs AB et AC qui vont nous permettre de calculer la normale du triangle
        v1 = creer_vecteur([pointA, pointB])
        v2 = creer_vecteur([pointA, pointC])
        #on cree la normale
        normale = calcul_normale(v1, v2)
    return normale

# 2 : l'illumination

#lumière diffuse
def vecteur_light(coor_point, coor_src_lum):
    '''
    cree un vecteur (appele vecteur 'light') qui va du pixel jusqu'a la source de lumiere
    entree : coor_point : les coordonnees du point (exemple : (x, y, z))
            coor_src_lum : coordonnees de la source de lumiere (exemple : (x, y, z))
    sortie : vec_light : coordonnees du vecteur light 
    '''
    #creation du vecteur light
    vecteur_l = creer_vecteur([coor_point, coor_src_lum])
    #nous voulons normaliser ce vecteur, nous allons donc diviser ce vecteur par sa norme
    #récupération des coordonnées du vecteur light
    x = vecteur_l[0]
    y = vecteur_l[1]
    z = vecteur_l[2]
    #calcul de la norme et du vecteur light
    norme_light = sqrt(x**2 + y**2 + z**2)
    x = x/norme_light
    y = y/norme_light
    z = z/norme_light
    vec_light = (x, y, z)
    return vec_light

def illumination(light, normale):
    '''
    calcul l'illumination d'un pixel en faisant le produit scalaire du vecteur
    light du pixel avec la norme du triangle auquel le pixel appartient
    entree : light : le vecteur light du pixel
            normale : le vecteur normale du triangle du pixel
    sortie : illumi : la valeur de l'illumination du pixel (compriser entre -1 et 1) 
    '''
    #récupération des coordonnées du vecteur light
    a = light[0]
    b = light[1]
    c = light[2]
    #récupération des coordonnées du vecteur normale
    d = normale[0]
    e = normale[1]
    f = normale[2]
    #calcul du produit scalaire entre les vecteurs light et normale avec la 
    #formule u.v = xx' + yy' + zz', et on prend le maximum entre 0 et la resultat 
    #car on ne prend pas compte de l'organisation du triangle
    illumi = max((a*d + b*e + c*f), 0)
    return illumi

# speculaire 
def vecteur_vue(coor_point, coor_camera):
    '''permet de calculer le vecteur vue 
    entree : coor_point : les coordonnees du point (exemple : (x, y, z))
            coor_camera : les coordonnees de la camera decrites dans la variable gloable 'c', un tableau de flotant
    sortie : vect : le vecteur vue (exemple : (x, y, z))
    '''
    vect =  vecteur_light(coor_point, coor_camera)
    return vect

def vecteur_H(vect_light, vect_vue):
    '''permet de calculer le vecteur H 
    entree : vect_light : le vecteur light (exemple : (x, y, z))
            vect_vue : le vecteur vue (exemple : (x, y, z))
    sortie : 
    '''
    #on calcul d'abord la somme des vecteurs light et vue
    #coor vect light
    xl = vect_light[0] #x light
    yl = vect_light[1]
    zl = vect_light[2]
    #coor vect vue
    xv = vect_vue[0] #x vue
    yv = vect_vue[1]
    zv = vect_vue[2]
    #somme 
    xs = xl+xv #x somme 
    ys = yl+yv
    zs = zl+zv
    #on calcule la norme de ce vecteur 
    norme_vect = sqrt(xs**2 + ys**2 + zs**2)
    #on normalise ce vecteur 
    x = xs/norme_vect
    y = ys/norme_vect
    z = zs/norme_vect
    vect_H = (x, y, z)
    return vect_H

def speculaire(vect_H, normale, alpha):
    '''
    calcule le coefficient specualire 
    entrees : vect_H : le vecteur H (exemple : (x, y, z))
            normale : la normale du triangle (exemple : (x, y, z))
            alpha : entier qui depend du materiaux, entier,  variable globale
    sortie : coeff_spec : le coefficient specualire 
    '''
    #on fait d'abord le produit scalaire des vecteurs H et de la normale
    xh = vect_H[0]
    yh = vect_H[1]
    zh = vect_H[2]
    xn = normale[0]
    yn = normale[1]
    zn = normale[2]
    #on calcul le produit scalaire 
    prod_scalaire = xh*xn + yh*yn + zh*zn
    #on met le produit scalaire a la puissance alpha 
    coeff_spec = prod_scalaire**alpha
    return coeff_spec

# 3 : cordonnees dans le monde apres les transformations 

def coor_monde_apres_transformation(triangles):  
    '''
    permet d'avoir les coordonnees des sommets du triangle dans le monde apres 
    les tranformations
    entree : un triangle qui a les memes caracteristiques qu'un triangles dans 
    la variable globale 'triangles'
    Sortie : coor_monde : les coordonnees des sommets du triangle dans le monde
    '''
    #on renprend les parametres pour la fonction local_vers_monde pour que cela soit plus visible 
    d = triangles[1] #[dx,dy,dz] Les coefficients de dilatation sur chaque axe
    r = triangles[2] #[ax,ay,az] Les angles de rotation autour de chaque axe
    t = triangles[3] #[tx,ty,tz] Les coefficients de translation sur chaque axe
    va = triangles[0][0][0] + (1,) #[x,y,z] Les coordonnées de base du point A
    vb = triangles[0][1][0] + (1,)  #[x,y,z] Les coordonnées de base du point B
    vc = triangles[0][2][0] + (1,)  #[x,y,z] Les coordonnées de base du point C
    #on recupere les cordonnees de x de la matrice cree par la fonction local_vers_monde
    xa = local_vers_monde(va, d, r, t)[0] 
    ya = local_vers_monde(va, d, r, t)[1]
    za = local_vers_monde(va, d, r, t)[2]
    pointA = (xa, ya, za)
    xb = local_vers_monde(vb, d, r, t)[0]
    yb = local_vers_monde(vb, d, r, t)[1]
    zb = local_vers_monde(vb, d, r, t)[2]
    pointB = (xb, yb, zb)
    xc = local_vers_monde(vc, d, r, t)[0]
    yc = local_vers_monde(vc, d, r, t)[1]
    zc = local_vers_monde(vc, d, r, t)[2]
    pointC = (xc, yc, zc)
    coor_monde = [pointA, pointB, pointC]
    return coor_monde

def retrouver_point(cf1, cf2, cf3, pointA, pointB, pointC):
    '''
    on recupere la position du point dans le monde apres les transformations 
    avec les coordonnees barycentriques
    entrees : cf1/cf2/cf3 : les coefficients barycentriques rensegnes dans ''calculer rgb''
    pointA/B/C : les points des sommets du triangle apres les transformations, en tuple 
    sortie : point : les coordonnees du point dans le monde 
    '''
    #on recupere les cordonnees du point A 
    a = pointA[0]
    b = pointA[1]
    c = pointA[2]
    #on recupere les cordonnees du point B 
    d = pointB[0]
    e = pointB[1]
    f = pointB[2]
    #on recupere les cordonnees du point C
    g = pointC[0]
    h = pointC[1]
    i = pointC[2]
    #on utilise les cf pour calculer les coordonnees du point dans le monde 
    x = cf1*a + cf2*d + cf3*g
    y = cf1*b + cf2*e + cf3*h
    z = cf1*c + cf2*f + cf3*i
    point = (x, y, z)
    return point

# 4 : la texture 

def applique_texture(cf1, cf2, cf3, texture):
    '''
    permet d appliquer la texture au triangle
    entree : cf1/2/3 :  les coefficients barycentriques du point 
    sortie : info_texture : array avec les couleurs RGB de la texture et le dtype
    '''
    width = texture.shape[0]
    height = texture.shape[1]
    #on recupere les coordonnees des sommets de la texture 
    #on recupere les cordonnees du point A : (a, b)
    a = 0
    b = 0
    #on recupere les cordonnees du point B : (c, d)
    c = width -1
    d = 0
    #on recupere les cordonnees du point C : (e, f)
    e = 0
    f = height -1
    #on utilise les cf pour calculer les coordonneesdu point dans le monde 
    x = int(cf1*a + cf2*c + cf3*e)
    y = int(cf1*b + cf2*d + cf3*f)
    info_texture = texture[x][y]
    return info_texture

# 5 : la sphere 

def base_sphere(cote, centre, modele):
    '''
    permet de calculer les sommets d'une pyramide equilaterale centree en (0,0,0)
    entree : cote : la taille du cote des triangles qui composent cette pyramide, int 
            centre : le centre de la puramide (ici (0,0,0))
            modele : le modele sur lequel on se base pour recreer la liste des triangles 
    sortie : liste_triangles :liste des 4 triangles qui composent la pyamide 
    '''
    #on cherche la hauteur de l'un des triangles equi de coté 3
    h = sqrt(cote**2 - 1.5**2)
    zD = h * 2/3 #le z de D 
    zBC = h * 1/3 #le z de B et de C 
    hp = sqrt(3**2 - h * 2/3) #la hauteur de la pyramide 
    yA = hp * 2/3 #le y de A 
    yBCD = hp * (-1/3) #le y de B,C,D
    #on recree les differents points qui composent le triangle
    A = (0., yA, 0.)
    B = (-cote/2, yBCD, zBC)
    C = (cote/2, yBCD, zBC)
    D = (0., yBCD, zD)
    #on va ensuite creer les vecteurs qui vont du centre aux sommets, on les normalise et on recree les coordonnees des sommets 
    #on recupere les vecteurs qui vont du centre au point 
    vectA = creer_vecteur([centre, A])
    vectA = norme_vect(vectA) #on normalise ce vecteur pour qu'il soit de norme 1
    vectB = creer_vecteur([centre, B])
    vectB = norme_vect(vectB)
    vectC = creer_vecteur([centre, C])
    vectC = norme_vect(vectC)
    vectD = creer_vecteur([centre, D])
    vectD = norme_vect(vectD)
    #on redefini les coordonnees, donc on retrouve les points D, E et F a partir du vecteur normalise 
    pointA = calcul_point(centre, vectA)
    pointB = calcul_point(centre, vectB)
    pointC = calcul_point(centre, vectC)
    pointD = calcul_point(centre, vectD)
    liste_triangles = [recree_triangle(modele, pointA, pointB, pointC), 
                       recree_triangle(modele, pointA, pointC, pointD),
                       recree_triangle(modele, pointC, pointB, pointD),
                       recree_triangle(modele, pointB, pointA, pointD)]
    return liste_triangles 

def sphere(triangle, centre_transfo):
    '''
    permet de diviser les triangles pour creer une liste de triangles qui forment une sphere
    entree : triangles: liste de triangels comme decrits dans la variable globale 
            centre : le centre de la sphere (ici (0, 0, 0)), en tuple 
    sortie : new_liste_triangles : le nouvelle liste avec tous les nouveaux triangles qui composent la sphere 
    '''
    modele = triangle[0] #on cree un modele pour nous aider a ajouter des triangles de la meme maniere dont ils sont decrits
    new_liste_triangles = [] #on cree une nouvelle liste ou on aura tous nos nouveaux triangles
    for i in range (0, len(triangle)): #pour chaque triangle
        #on recupere les points du triangle et on en cree d'autres pour former d'autres triangles
        pointA = triangle[i][0][0][0]
        pointB = triangle[i][0][1][0]
        pointC = triangle[i][0][2][0]
        pointD = point_millieu(pointA, pointB)
        pointE = point_millieu(pointA, pointC)
        pointF = point_millieu(pointB, pointC)
        #on met les points D, E et F a une distance de 1 du centre 
        #on recupere les vecteurs qui vont du centre au point 
        vectD = creer_vecteur([centre_transfo, pointD])
        vectD = norme_vect(vectD) #on normalise ce vecteur pour qu'il soit de norme 1
        vectE = creer_vecteur([centre_transfo, pointE])
        vectE = norme_vect(vectE)
        vectF = creer_vecteur([centre_transfo, pointF])
        vectF = norme_vect(vectF)
        #on redefini les coordonnees, donc on retrouve les points D, E et F a partir du vecteur normalise 
        pointD = calcul_point(centre_transfo, vectD)
        pointE = calcul_point(centre_transfo, vectE)
        pointF = calcul_point(centre_transfo, vectF)
        #on recree 4 triangles a partir de ce seul triangle pour recreer une liste de triangles 
        beta = modele 
        beta = recree_triangle(beta, pointD, pointB, pointF)
        gamma = modele 
        gamma = recree_triangle(gamma, pointE, pointF, pointC)
        delta = modele
        delta = recree_triangle(delta, pointE, pointD, pointF)
        epsilon = modele 
        epsilon = recree_triangle(epsilon, pointD, pointE, pointA)
        #on ajoute ces 4 triangles a la liste de tous les triangles qui composent la sphere
        new_liste_triangles += [beta, gamma, delta, epsilon]
    return new_liste_triangles
        
def point_millieu(point_a, point_b):
    '''
    trouve le point du millieu entre deux point a l'aide de leurs coordonnees
    entree : point_a, point_b : les deux point dont on veux trouver le millieu, en tuple
    sortie : millieu : les coordonnees du point du millieu, en tuple 
    '''
    #on recupere les coordonnees des points 
    xa = point_a[0]
    ya = point_a[1]
    za = point_a[2]
    xb = point_b[0]
    yb = point_b[1]
    zb = point_b[2]
    millieu = ((xa+xb)/2, (ya+yb)/2, (za+zb)/2)
    return millieu

def norme_vect(vecteur):
    '''
    calcul la norme d'un vecteur et divise le vecteur par sa normale (donc normalise un vecteur)
    entree : un vecteur, en tuple (exemple : (x, y, z))
    sortie : (x, y, z), les coordonnees du nouveau vecteur 
    '''
    x_vect = vecteur[0]
    y_vect = vecteur[1]
    z_vect = vecteur[2]
    #calcul de la norme de ce vecteur
    norme_vec = sqrt(x_vect**2 + y_vect**2 + z_vect**2)
    #division du vecteur par sa norme
    x = x_vect/norme_vec
    y = y_vect/norme_vec
    z = z_vect/norme_vec
    return (x, y, z)

def calcul_point(centre, vecteur):
    '''
    calcul les coordonnees du point avec le point de depart et le vecteur 
    entrees: centre : le centre de la sphere, (ici (0, 0, 0)), en tuple
            vecteur : le vecteur qui va du centre a l'un des sommet du triangle 
            choisi, en tuple (exemple : (x, y, z))
    sortie : (x, y, z), les coordonnees du nouveau point eloigné de 1 du centre
    '''
    #pour la sphere le depart de tous les vecteurs est a 0, donc on prend simplement les cordonnees du vecteur pour retrouver celles du point
    x = vecteur[0]
    y = vecteur[1]
    z = vecteur[2]
    return (x, y, z)

def recree_triangle(modele, point_a, point_b, point_c):
    '''
    recree un triangle comme decrit dans la variable globale 'triangles'
    entree : modele : le modele sur lequel on se base pour decrire le nouveau triangle 
            point_a/b/c : les deifferents points qui composent ce triangle 
    sortie : modele : le triangle que l'on a choisi mis sur le meme modele que 
    les triangles definis sans la variable globale 'triangles''
    '''
    res = modele.copy()
    res[0] = modele[0].copy()
    res[0][0] = modele[0][0].copy()
    res[0][1] = modele[0][1].copy()
    res[0][2] = modele[0][2].copy()
    res[0][0][0] = point_a
    res[0][1][0] = point_b
    res[0][2][0] = point_c
    return res

def creer_sphere(triangle, centre_transfo, omega):
    '''
    cree la liste omega fois pour former la sphere, donc une liste avec tous les triangles de la sphere 
    entrees : triangles : la liste des triangles dans la figure, comme decrits dans la variable globale 
            centre : le centre de la shpere (ici (0, 0, 0)), en tuple
            omega : le nombre de fois ou l'on veut diviser les triangles
    sortie : new_liste : la nouvelle liste de tous les triangles qui composent da sphere 
    '''
    new_liste = sphere(triangle, centre_transfo)
    for i in range (0, omega) :
        new_liste = sphere(new_liste, centre_transfo)
    return new_liste 

#pour mettre les normales aux sommets
def normale_point_sphere(cf1, cf2, cf3, normaleA, normaleB, normaleC):
    '''
    calcul (pour chaque point qui composent la sphere) sa normale avec les coordonnees barycentriques
    entrees : cf1, cf2, cf3 : les coeff barycentriques du point 
            normaleA, normaleB, normaleC : les normales des 3 sommets qui composent le triangle duquel vient le point 
    sorties : normale_point : la normale du point en tuple (ex : (x, y, z))
    '''
    #on recupere les cordonnees de la normale du point A 
    a = normaleA[0]
    b = normaleA[1]
    c = normaleA[2]
    #on recupere les cordonnees du point B 
    d = normaleB[0]
    e = normaleB[1]
    f = normaleB[2]
    #on recupere les cordonnees du point C
    g = normaleC[0]
    h = normaleC[1]
    i = normaleC[2]
    #on applique les coefficients barycentriques
    x = cf1*a + cf2*d + cf3*g
    y = cf1*b + cf2*e + cf3*h
    z = cf1*c + cf2*f + cf3*i
    normale = (x, y, z)
    #on normalise ensuite cette normale
    normale_point = norme_vect(normale)
    return normale_point

# 6 : les ombres

def retrouver_pixel_img_lum(point, cam_lum): 
    '''
    permet de retrouver la position du point passe en paramètre sur l'image vue par la camera de la lumiere
    entrees : point : un tuple qui donne les cordonnees monde du point (ex : (x, y, z))
            image_lumiere : les z_buffer de tous les points vus par la lumiere, un tableau de tableau
            cam_lum : la camera mise a la place de la lumiere comme decrite dans la variabmle globale
    sortie : [x, y] les cordonnees du pixel sur l'image, un tableau 
    '''
    point = point+(1,)
    #on recupere la position du point sur la camera de la lumiere
    coor_pt_raster = procedure_local_grille2(point,cam_lum[0],cam_lum[1],cam_lum[2],cam_lum[3],cam_lum[4],cam_lum[5],image[0],image[1])
    x = coor_pt_raster[0]
    y = coor_pt_raster[1]
    z = coor_pt_raster[2]
    return [x, y, z]
    
def est_visible(point, image_lumiere, cam_lum):
    '''
    permet de savoir si un point est vu ou non dans l'image vu par la camera de la lumiere
    entrees: point : les coordonnees du point teste (un tuple)
            image_lumiere : les z_buffer de tous les points vus par la lumiere, un tableau de tableau
            cam_lum : la camera de la lumiere, un tableau comme decit dans la variable globale
    sortie : visible : un booleen qui dit si oui ou non le point est visible par la lumiere 
    '''
    pixel = retrouver_pixel_img_lum(point, cam_lum) #on a les coordonnees x, y et z du pixel sur lequel le point dans le monde est projete 
    z_buffer_pt = pixel[2] #le z_buffer du point
    x = pixel[0]
    y = pixel[1]
    #print(pixel)
    if x < 0: #si le point n'est pas dans le cadre de la camera 
        x = 0
    if y < 0: 
        y = 0
    z_buffer_pix = image_lumiere[y][x] #le z_buffer du pixel qui est deja vu par l'image à l'endroit ou est projete le point 
    #print(z_buffer_pix)
    if z_buffer_pt + 0.01 >= z_buffer_pix: 
        visible = True 
    else: 
        visible = False
    return visible

def procedure_local_grille2(point_monde,dc,rc,tc,fov,proche,lointain,largeur,hauteur):
    """
    Transformation totale du point monde vers l'espace raster (presque exact a la fonction procedure_local_grille)
    entrees : point_monde : le point dans le monde, un tuple
            dc : la transformation dilatation de la camera, tableau d'entier 
            rc : la transformation rotation de la camera, tableau d'entier 
            tc : la transformation translation de la camera, tableau d'entier 
            fov : l'angle de vue, un entier 
            proche : distance avec le plan proche (znear), entier relatif
            lointain : distance avec le plan lointain (zfar), entier relatif
            largeur : la largeur de l'image de la camera, un entier 
            hauteur : la heuteur de l'image de la camera, un entier
    sortie : coords :  les cordonnees du point sur l'image de la camera 
    """
    
    point_camera = monde_vers_camera(point_monde,dc,rc,tc)
    (point_ecran, w) = camera_vers_ecran(point_camera,fov,proche,lointain)
    point_normalise = point_ecran
    point_normalise[0] = (point_normalise[0] + 1.) / 2.
    point_normalise[1] = (point_normalise[1] + 1.) / 2.
    point_grille = normalise_vers_grille(point_normalise, largeur, hauteur)
    x = point_grille[0]
    y = point_grille[1]
    coords = (int(x),int(y),point_grille[2])
    return coords  #Ajout d'un index et de la couleur necessaire
                                    #à la visualisation
    
    
#-----------------------------------------------------------------------------

# Algorithme final

def rasterisation(triangles,c,image, coor_src_lum):
    """
    triangles : [[[[(x,y,z),index,couleur],[(x,y,z),index,couleur],[(x,y,z),index,couleur]],dp,rp,tp],triangle2...]
    c (camera) : [dc,rc,tc,fov,proche,lointain]
    image : [largeur,hauteur,fond,fichier_sortie]
    coor_src_lum : coordonnees de la source de lumiere sous forme de tuple (x, y, z)
    """
    points_p = []
    triangles_p = []
    l = image[0]
    h = image[1]
    fond = image[2]
    fichier = image[3]
    #Pour chaque triangle
    ws = []
    for t in triangles:
        #Pour chaque point
        for p in t[0]:
            point = p[0]+(1,)
            (point, w) = procedure_local_grille(point,p[1],p[2],t[1],t[2],t[3],c[0],c[1],c[2],c[3],c[4],c[5],l,h)
            points_p.append(point)
            ws.append(w)
        #Ajout du nouveau point
        triangles_p.extend([points_p])
        points_p = []
    liste_pixels_t = []
    #Pour chaque triangle projete
    for t in range (0, len(triangles_p)):
        coor_monde = coor_monde_apres_transformation(triangles[t])
        normale = normale_triangles(triangles[t])
        liste_pixels_t.append(remplir(triangles_p[t][0],triangles_p[t][1],triangles_p[t][2], normale, triangles[t], l, h, ws[3*t], ws[3*t+1], ws[3*t+2], coor_monde))
    #Initialisation z-buffer global car reutilise dans plusieurs fonctions
    global z_buffer
    z_buffer = initialisation_z_buffer(l,h)
    creer_image(l,h,fond,fichier)
    #Test z-buffer
    for p in liste_pixels_t:
        rendu_z_buffer(l,h,p,fichier,z_buffer)
    
    return "Rendu terminé"


#parametres et differentes fonctionnalites du code

#une liste de triangles avec un seul trinangle
triangles = [
        # 1 triangle = 3 points + une transformation
        [
            # 3 points: position, indice, couleur (RGB)
            [[(0., 0., 0.),0,[0, 255, 0]],
            [(1.,0.,0.),1,[0, 0, 255]],
            [(0.,1.,0.),2,[255, 0, 0]]
             ],
         # transformation : dilatation, rotation, translation
         [1., 1., 1.],[0., 0., 0.],[0., 0., -2.]]
        ]

# camera : tranformation + "paramètres de vue"
c = [
        # tranformation
        [1., 1., 1.],[0., 0., 0.],[0., 0., 0.],
        # paramètres : angle de vue, znear, zfar (bornes de profondeurs de la "boîte" finale)
        90.,-0.1,-100.
        ]
# image: largeur, hauteur, couleur du fond (RGB), nom du fichier de sortie (garder le .ppm)
image= [480, 480, [0, 0, 0], "testwk_fix.ppm"]

#les cordonnees de la source de lumiere (x,y,z)
coor_src_lum = (1., 0.5, -0.5)
#(0., 2., 0.) coordonnees parfaites pour un ombre projetee sur le cube 

#on ajoute une texture
texture = matplotlib.image.imread("eau.jpg")
#on demande si on utilise la texture : True oui, False non
ask_texture = True

#pour les speculaires
alpha = 10 #float qui depend du materiaux sur lequel la lumiere se reflette 
coor_camera = c[2] #coordonnees de la camera 

#pour la sphere 
#modele pour la sphere, sert notamment a avoir les differentes transformations 
modele = [
    # 3 points: position, indice, couleur (RGB)
    [[(0., 0., 0.),0,[255, 0, 0]],
    [(1.,0.,0.),1,[0, 255, 0]],
    [(0.,1.,0.),2,[0, 0, 255]]
     ],
 # transformation : dilatation, rotation, translation
 [1., 1., 1.],[0., 0., 0.],[0., 0., -3.]]
#autres donnees pour la sphere 
cote = 2 #la longueur des cotes des triangles qui omposent la pyramide a base triangulaire equilaterale qui sert de base pour la sphere 
centre = (0, 0, 0) #le centre de la sphere 
centre_transfo = modele[3] #le centre le la sphere apres les transformations 
omega = 1 #le nombre de fois ou on divise les triangles pour creer la sphere 
base_sp = base_sphere(cote, centre, modele) #la base de la sphere (la pyramide de base)
la_sphere = creer_sphere(base_sphere(cote, centre, modele), centre, omega) #la liste des triangles qui composent la sphere 
ask_lisse_sphere = False #on demande si l'objet est une sphere et si on veut la 'lisser' (donc mettre les normales aux sommets)

#pour les ombres projetees 
ask_ombre = False #on demande si on veut une ombre
# camera de la lumière : tranformation + "paramètres de vue"
cam_lum = [# tranformation.
        [1., 1., 1.],[coor_src_lum[0], coor_src_lum[1], coor_src_lum[2]], [90., 0., 0],
        # paramètres : angle de vue, znear, zfar (bornes de profondeurs de la "boîte" finale)
        90.,-0.1,-100.]
# lancement du programme pour avoir la liste des z_buffer des points 'vu' par la lumiere, ce qui va nou saider a creer l'ombre
image_lumiere = rasterisation_ombre(cube, cam_lum, image)


# différentes figures en 3D 
#deux triangles qui forment un carre
carre = [
        # 1 triangle = 3 points + une transformation
        [
            # 1er triangle rouge
            [[(0., 0., 0.),0,[255, 0, 0]],
            [(1.,0.,0.),1,[255, 0, 0]],
            [(0.,1.,0.),2,[255, 0, 0]]
             ],
         # transformation : dilatation, rotation, translation
         [1., 1., 1.],[0., 0., 0.],[0., 0., -1.]],
        [
            # 2eme triangle vert
            [[(1., 1., 0.),0,[0, 255, 0]],
            [(0.,1.,0.),1,[0, 255, 0]],
            [(1.,0.,0.),2,[0, 255, 0]]
             ],
         # transformation : dilatation, rotation, translation
         [1., 1., 1.],[0., 0., 0.],[0., 0., -1.]]
        ]

#plusieurs triangles qui froment un cube avec un sol 
cube = [
        # 1 triangle = 3 points + une transformation
        [
            # 1er triangle rouge
            [[(0., 0., 0.),0,[255, 0, 0]],
            [(1.,0.,0.),1,[255, 0, 0]],
            [(0.,1.,0.),2,[255, 0, 0]],
             ],
         # transformation : dilatation, rotation, translation
         [0.2, 0.2, 0.2],[50., 150., 0.],[0., 0., -1.]],
        [
            # 2eme triangle vert
            [[(1., 1., 0.),0,[0, 255, 0]],
            [(0.,1.,0.),1,[0, 255, 0]],
            [(1.,0.,0.),2,[0, 255, 0]],
             ],
         # transformation : dilatation, rotation, translation
         [0.2, 0.2, 0.2],[50., 150., 0.],[0., 0., -1.]],
        [
            # 3e triangle bleu
            [[(1., 1., 0.),0,[0, 0, 255]],
            [(1.,0.,0.),1,[0, 0, 255]],
            [(1.,1.,-1.),2,[0, 0, 255]],
             ],
         # transformation : dilatation, rotation, translation
         [0.2, 0.2, 0.2],[50., 150., 0.],[0., 0., -1.]],
        [
            # 4e triangle jaune
            [[(1., 0., 0.),0,[255, 255, 0]],
            [(1.,0.,-1.),1,[255, 255, 0]],
            [(1.,1.,-1.),2,[255, 255, 0]],
             ],
         # transformation : dilatation, rotation, translation
         [0.2, 0.2, 0.2],[50., 150., 0.],[0., 0., -1.]],
        [
            # 5e triangle cyan
            [[(1., 1., 0.),0,[0, 255, 255]],
            [(1.,1.,-1.),1,[0, 255, 255]],
            [(0.,1.,0.),2,[0, 255, 255]],
             ],
         # transformation : dilatation, rotation, translation
         [0.2, 0.2, 0.2],[50., 150., 0.],[0., 0., -1.]],
        [
            # 6e triangle rose
            [[(1., 1., -1.),0,[255, 0, 255]],
            [(0.,1.,-1.),1,[255, 0, 255]],
            [(0.,1.,0.),2,[255, 0, 255]],
             ],
         # transformation : dilatation, rotation, translation
         [0.2, 0.2, 0.2],[50., 150., 0.],[0., 0., -1.]],
        [
            # 7e triangle violet
            [[(0., 0., 0.),0,[120, 0, 120]],
            [(1.,0.,-1.),1,[120, 0, 120]],
            [(1.,0.,0.),2,[120, 0, 120]],
             ],
         # transformation : dilatation, rotation, translation
         [0.2, 0.2, 0.2],[50., 150., 0.],[0., 0., -1.]],
        [
            # 8e triangle orange
            [[(0., 0., 0.),0,[255, 1, 0]],
            [(0.,0.,-1.),1,[255, 1, 0]],
            [(1.,0.,-1.),2,[255, 1, 0]],
             ],
         # transformation : dilatation, rotation, translation
         [0.2, 0.2, 0.2],[50., 150., 0.],[0., 0., -1.]],
        [
            # 9e triangle blanc
            [[(0., 0., 0.),0,[255, 255, 255]],
            [(0.,1.,-1.),1,[255, 255, 255]],
            [(0.,0.,-1.),2,[255, 255, 255]],
             ],
         # transformation : dilatation, rotation, translation
         [0.2, 0.2, 0.2],[50., 150., 0.],[0., 0., -1.]],
        [
            # 10e triangle gris
            [[(0., 0., 0.),0,[100, 100, 100]],
            [(0.,1.,0.),1,[100, 100, 100]],
            [(0.,1.,-1.),2,[100, 100, 100]],
             ],
         # transformation : dilatation, rotation, translations
         [0.2, 0.2, 0.2],[50., 150., 0.],[0., 0., -1.]],
        [
            # 11e triangle) lavende
            [[(0., 1., -1.),0,[195, 147, 215]],
            [(1.,0.,-1.),1,[195, 147, 215]],
            [(0.,0.,-1.),2,[195, 147, 215]],
             ],
         # transformation : dilatation, rotation, translation
         [0.2, 0.2, 0.2],[50., 150., 0.],[0., 0., -1.]],
        [
            # 12e triangle vert pastel
            [[(1., 0., -1.),0,[176, 242, 182]],
            [(0.,1.,-1.),1,[176, 242, 182]],
            [(1.,1.,-1.),2,[176, 242, 182]],
             ],
        # transformation : dilatation, rotation, translation
         [0.2, 0.2, 0.2],[50., 150., 0.],[0., 0., -1.]],
        [
            [[(1., -1., -1.),0,[255, 255, 255]],
            [(-1.,-1.,0.),1,[255, 255, 255]],
            [(1.,-1.,0.),2,[255, 255, 255]],
             ],
         # transformation : dilatation, rotation, translation
         [1., 1., 1.],[0., 0., 0.],[0., 0., -1.]],

        [
            [
            [(-1.,-1.,0.),0,[255, 255, 255]],
            [(1.,-1.,-1.),1,[255, 255, 255]],
            [(-1., -1., -1.),2,[255, 255, 255]],
             ],
         # transformation : dilatation, rotation, translation
         [1., 1., 1.],[0., 0., 0.],[0., 0., -1.]],
        ]

#plusieurs triangles qui forment une pyramide a base triangulaire quelconque
pyramide = [
        # 1 triangle = 3 points + une transformation
        [
            # 1er triangle rouge
            [[(0., 0., 0.),0,[255, 0, 0]],
            [(1.,0.,0.),1,[255, 0, 0]],
            [(0.,1.,0.),2,[255, 0, 0]]
             ],
         # transformation : dilatation, rotation, translation
         [1., 1., 1.],[0., 160., 0.],[0., 0., -3.]],
        [
            # 2e triangle vert
            [[(0., 0., 0.),0,[0, 255, 0]],
            [(0.,1.,0.),1,[0, 255, 0]],
            [(0.,0.5,1.),2,[0, 255, 0]]
             ],
         # transformation : dilatation, rotation, translation
         [1., 1., 1.],[0., 160., 0.],[0., 0., -3.]],
        [
            # 3e triangle bleu
            [[(0., 1., 0.),0,[0, 0, 255]],
            [(1.,0.,0.),1,[0, 0, 255]],
            [(0.,0.5,1.),2,[0, 0, 255]]
             ],
         # transformation : dilatation, rotation, translation
         [1., 1., 1.],[0., 160., 0.],[0., 0., -3.]],
        [
            # 4e triangle jaune
            [[(0., 0., 0.),0,[255, 255, 0]],
            [(0.,0.5,1.),1,[255, 255, 0]],
            [(1.,0.,0.),2,[255, 255, 0]]
             ],
         # transformation : dilatation, rotation, translation
         [1., 1., 1.],[0., 160., 0.],[0., 0., -3.]]
        ]


# lancement du programme
rasterisation(carre, c, image, coor_src_lum)