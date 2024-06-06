# coding: utf-8
#code ecrit pas Cassiopée Gossin - L1 CMI informatique (2023) et Raphael Tournafond - L1 CMI informatique (2017)

#Importations necessaires
import numpy as np
from numpy.linalg import inv
from math import cos, sin, tan, pi, floor, inf, sqrt
#from PIL import Image #Importation bibliothèque image (TP1-Math202)

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


def remplir(s1,s2,s3, triangle, width, height, w1, w2, w3):
    """
    Renvoie la liste des pixels ainsi que leur couleur
    s = ((x,y,z),i,(r,g,b))
    """
    
    o = ordre(s1,s2,s3)
    liste = liste_pixels_coeffs(o[0][0],o[1][0],o[2][0],width,height,w1,w2,w3)
    return liste #calculer_rgb(s1[2],s2[2],s3[2],liste, normale, triangle, coor_monde)

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
        x = i[0]
        y = i[1]
        z = i[2]
        #c = i[1]
        #tests z-buffer
        if not cache(z_buffer,(x,y,z)):
            z_buffer = modifier_z_buffer(z_buffer,(x,y,z))
            data[3 + x + y * l] = str(int(z_buffer[y][x]*100)) + " " + str(int(z_buffer[y][x]*100)) + " " + str(int(z_buffer[y][x]*100)) + "\n"
            #im.setPixel(x,y,c)

    #Sauvegarde de l'image
    with open(fichier,"w") as image:
        image.writelines(data)
    return None

# Algorithme final

def rasterisation_ombre(triangles,c,image):
    """
    triangles : [[[[(x,y,z),index,couleur],[(x,y,z),index,couleur],[(x,y,z),index,couleur]],dp,rp,tp],triangle2...]
    c (camera) : [dc,rc,tc,fov,proche,lointain]
    image : [largeur,hauteur,fond,fichier_sortie]
    coor_src_lum : coordonnees de la source de lumiere sous forme de tuple (x, y, z)
    """
    points_p = []
    print()
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
        #coor_monde = coor_monde_apres_transformation(triangles[t])
        #normale = normale_triangles(triangles[t])
        liste_pixels_t.append(remplir(triangles_p[t][0],triangles_p[t][1],triangles_p[t][2], triangles[t], l, h, ws[3*t], ws[3*t+1], ws[3*t+2]))
    #Initialisation z-buffer global car reutilise dans plusieurs fonctions
    global z_buffer
    z_buffer = initialisation_z_buffer(l,h)
    creer_image(l,h,fond,fichier)
    #Test z-buffer
    for t in liste_pixels_t: #pour chauqe triangle
        rendu_z_buffer(l,h,t[0],fichier,z_buffer)
    return z_buffer

"""
triangles = [
        # 1 triangle = 3 points + une transformation
        [
            # 3 points: position, indice, couleur (RGB)
            [[(0., 0., 0.),0,[0, 255, 0]],
            [(1.,0.,0.),1,[0, 0, 255]],
            [(0.,1.,0.),2,[255, 0, 0]]
             ],
         # transformation : dilatation, rotation, translation
         [1., 1., 1.],[0., 0., 0.],[0., 0., -1.]]
        ]

coor_src_lum = (0., 2., -0.2)


# camera : tranformation + "paramètres de vue"
c = cam_lum = [# tranformation.
        [1., 1., 1.],[coor_src_lum[0], coor_src_lum[1], coor_src_lum[2]], [90., 0., 0],
        # paramètres : angle de vue, znear, zfar (bornes de profondeurs de la "boîte" finale)
        90.,0.1,100.]
"""
# image: largeur, hauteur, couleur du fond (RGB), nom du fichier de sortie (garder le .ppm)
image= [480, 480, [0, 0, 0], "test_ombre_2.ppm"]

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
"""
# lancement du programme
rasterisation_ombre(cube, c, image)
"""