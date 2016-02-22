# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
from sklearn.cluster import KMean

# Crée une projection centrée sur San Francisco
frame = ((-122.52, 37.68), (-122.38,37.82))
map = Basemap(
        llcrnrlon=frame[0][0], llcrnrlat=frame[0][1],
        urcrnrlon=frame[1][0], urcrnrlat=frame[1][1],
        epsg=3493)
# Charge le plan associé à la projection
map.arcgisimage(
        service='ESRI_StreetMap_World_2D',
        xpixels = 1000, verbose= True)
latitudes = np.array([-122.425891675136 ,-122.425891675136]); longitudes = [37.7745985956747, 37.7745985956747]
couleurs = ['r', 'r' ]; rayons =[0.1, 0.1] 
# Convertit (long,lat) en position image
(X,Y)=map(longitudes, latitudes)
# Appel classique aux fonctions matplotlib
plt.scatter(X, Y, rayons, marker = 'o', color = couleurs,
        alpha = 0.4)
# Lance l’affichage
plt.show()
