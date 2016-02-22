# -*- coding: utf8 -*-
import pandas as pd
import numpy as np

# Création d'un "DataFrame"
#data = {
#"site"   : [ "Musée d'Art Moderne", "Musée du Louvres", "Musée d'Orsay", "Centre Pompidou", "Centre Pompidou", "Musée des Beaux-Arts" ],
#"type"   : [ "Contemporain", "Classique", "Impressionisme", "Contemporain", "Contemporain", "Classique" ],
#"ville"  : [ "Strasbourg", "Paris", "Paris", "Paris", "Metz", "Nancy" ],
#"visite" : [ 2011, 2012, 2012, 2012, 2015, 2011 ]
#}
#frame = pd.DataFrame(data, index = ["Str1", "Par1", "Par2", "Par3", "Mtz1", "Ncy1"])




# Sélections de colonnes et de lignes
print(frame[[1,2]].ix[1:3])
print(frame[['site','ville']].ix['Str1':'Mtz1'])
sample = frame.take(np.random.randint(0, len(frame), size = 2))

# Ajout, suppression de lignes, de colonnes
frame['nbr'] = [2,2,3,3,4,1,]
del frame['nbr']

frame.index.drop('Mtz1')

# Statistiques
print(frame.describe())
print(frame['visite'].mean())

# Histogrammes et discrétisation

pd.cut(frame.visite,2)
frame['periode']=pd.cut(frame.visite,2)

# Aggrégation
groups = frame.groupby(frame['type'])
print(groups.mean())

print(groups.visite.agg(lambda g : (g.min(), g.max())))

# Nettoyage et préparation
frame.drop_duplicates()
frame['code'] = frame['type'].map({ "Contemporain": 1, "Classique" : 2 })
frame = frame.dropna()

# Jointure
frame2 = pd.DataFrame({
"ville"  : [ "Strasbourg", "Paris","Metz", "Nancy" ],
"dept"   : [ 67, 75, 57, 54 ]
})
frame3 = pd.merge(frame, frame2, on='ville')
