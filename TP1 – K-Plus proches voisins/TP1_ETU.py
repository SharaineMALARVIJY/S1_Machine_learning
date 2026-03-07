# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 13:55:33 2025

@author: MALARVIJY Sharaine 21206543
"""

from matplotlib import pyplot as plt
from PIL import Image
from skimage.color import rgb2gray
import numpy as np
from display import plotHistoClasses, plotGallery
import sklearn.model_selection as sk
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# %%#############################################################################
# I Chargement des données

print("I Chargement des données")
# I.a. Chargement de la base
[X, y, name]=np.load('TP1.npy', allow_pickle=True)


plotGallery(X)
plotHistoClasses(y)

print("Taille des images", X.shape[1:])
print("Nombre d’images :", y.size)
print("Nombre de classes", len(name)) #j'ai trouvé ca aussi np.unique(y).size

for i in range(len(name)):
    print("Classe :", i, "Nombre exemples :", np.where(y == i)[0].shape[0], name[i])
print("Les classes ne sont pas equiprobable")



# I.b. Partitionnement de la base d’apprentissage

X_train, X_test, y_train, y_test = sk.train_test_split(X, y, random_state=543)


print("")
print("Nombre d’images X_train :", X_train.shape[0])
print("Nombre d’images X_test :", X_test.shape[0])
print("Nombre d’images y_train :", y_train.size)
print("Nombre d’images y_test :", y_test.size)
print("")
print("Dimensions X_train :", X_train.shape)
print("Dimensions X_test :", X_test.shape)
print("Dimensions y_train :", y_train.shape)
print("Dimensions y_test :", y_test.shape)

# %%#############################################################################
# II Prétraitement des données

# II.a. Redimensionnement des données
print("")
print("II Redimensionnement des données")
n=2914

X_train = X_train.reshape(X_train.shape[0], n)
X_test = X_test.reshape(X_test.shape[0], n)

print("Dimensions X_train :", X_train.shape)
print("Dimensions X_test :", X_test.shape)

# II.b. Mise en forme des données pour la classification

scaler = StandardScaler()

scaler_X_train = scaler.fit_transform(X_train)
scaler_X_test = scaler.transform(X_test)

print('La mise en forme des données consiste à transformer chaque caractéristique pour qu’elle ait une moyenne nulle et un écart-type égal à 1')

print("Moyenne X_train :", np.mean(np.mean(scaler_X_train, axis=1)))
print("Moyenne X_test :", np.mean(np.mean(scaler_X_test, axis=1)))

print("Ecart-type X_train :", np.std(scaler_X_train))
print("Ecart-type X_test :", np.std(scaler_X_test))

# %%#############################################################################
# III Classification par les KPPV
# III.c. Classifieur 1PPV 
print("")
print("III Classification par les KPPV")
PPV1 = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
PPV1.fit(scaler_X_train, y_train)
prediction = PPV1.predict(scaler_X_test)
print("Prédiction :", name[prediction[0]])
print("Réel :", name[y_test[0]])

conf_matrix = confusion_matrix(y_test, prediction)

accuracy = 0
for i in range(len(prediction)):
    if y_test[i]==prediction[i]:
        accuracy += 1
accuracy /= len(prediction)
print("Accuracy de sklearn :", accuracy_score(y_test, prediction))
print("Accuracy fait main  :", accuracy)

print("La matrice de confusion compare les classes réelles (lignes) aux classes prédites (colonnes)")

print(conf_matrix)
print("La somme de la matrice de confusion vaut", np.sum(conf_matrix))

for i in range(len(name)):
    print("Somme de la ligne", i,":", np.sum(conf_matrix[i]))

print('La somme totale des cases est nombre total d’exemples test')
print('La somme d’une ligne est nombre d’exemples réels de cette classe')

print("Les classes ne sont pas equiprobable dans la base de test")


# III.d. Classifieur KPPV

def KPPV(K, X_train, y_train, X_test):
    KPPV = KNeighborsClassifier(n_neighbors=K, metric='euclidean')
    KPPV.fit(X_train, y_train)
    prediction = KPPV.predict(X_test)
    return prediction

list_prediction = []
nb_point = range(1, 50)
for i in nb_point:
    prediction = KPPV(i, scaler_X_train, y_train, scaler_X_test)
    acc = accuracy_score(y_test, prediction)
    list_prediction.append(acc)

k_voisin_optimal = np.argmax(list_prediction) + 1

plt.figure(figsize=(8,6))
plt.plot(nb_point, list_prediction, marker="o")
plt.xlabel("Nombre de voisins (K)")
plt.ylabel("Accuracy")
plt.title("Évolution de l'accuracy en fonction de K")
plt.grid(True)
plt.show()

def traitement_image(im_path):
    I = Image.open(im_path)

    I_resized = I.resize((47,62))
    I = np.array(I_resized)
    I = rgb2gray(I) 

    I_reshape = I.reshape(1, 2914)
    I_reshape *= 255
    I_scaler = scaler.transform(I_reshape)
    print("Image :", im_path[9:-9])
    return I_scaler

list_im = ["Image_TP/Bush_reca.jpg", "Image_TP/Blair_reca.jpg", "Image_TP/Powell_reca.jpg"]
for i in list_im:
    I_scaler = traitement_image(i)
    y_pred = KPPV(k_voisin_optimal, scaler_X_train, y_train, I_scaler)
    print(name[y_pred])
