#%% Sharaine MALARVIJY 21206543

import numpy as np
import matplotlib.pyplot as plt
# from fn import norm1, norm2
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from scipy.stats import multivariate_normal, norm
from PIL import Image

DATA_PATH = 'Data'

image_ext = ['.jpg']
images =  [os.path.join(DATA_PATH, f) for f in os.listdir(DATA_PATH) if f.endswith(".jpg")]

X_train=np.empty((0, 2))
y_train=[]
for name in images[0:-4]:
    I = Image.open(name)
    I = I.convert('YCbCr')
    I = np.array(I)
    I = np.reshape(I[:,:,1:3],[I.shape[0]*I.shape[1],2])
    X_train = np.concatenate((X_train, I), axis=0)
    fichier_png= os.path.splitext(name)[0] + ".png"
    GT = Image.open(fichier_png)
    GT = np.array(GT)
    GT = np.reshape(GT[:,:,1]/255,[GT.shape[0]*GT.shape[1]])
    y_train = np.concatenate((y_train, GT), axis=0)


X_test=np.empty((0, 2))
y_test=[]
for name in images[-4:]:
    I = Image.open(name)
    I = I.convert('YCbCr')
    I = np.array(I)
    I = np.reshape(I[:,:,1:3],[I.shape[0]*I.shape[1],2])
    X_test = np.concatenate((X_test, I), axis=0)
    fichier_png = os.path.splitext(name)[0] + ".png"
    GT = Image.open(fichier_png)
    GT = np.array(GT)
    GT = np.reshape(GT[:,:,1]/255,[GT.shape[0]*GT.shape[1]])
    y_test = np.concatenate((y_test, GT), axis=0)

X_train = X_train.astype('float64')
X_test = X_test.astype('float64')

X_train, _, y_train, _ = train_test_split(X_train, y_train, train_size=1/1000, random_state=42)
X_test, _, y_test, _ = train_test_split(X_test, y_test, train_size=1/1000, random_state=42)

##############################################################################
# I.	Chargement et visualisation des données

#Pixel peau
Peau_Train = X_train[np.where(y_train==1),:]
Peau_Train = np.reshape(Peau_Train,(Peau_Train.shape[1],Peau_Train.shape[2] ))
#Pixel non peau
Nonpeau_Train = X_train[np.where(y_train==0),:]
Nonpeau_Train = np.reshape(Nonpeau_Train,(Nonpeau_Train.shape[1],Nonpeau_Train.shape[2] ))


plt.plot(Nonpeau_Train[:,0], Nonpeau_Train[:,1], '.b', label='Non peau')
plt.plot(Peau_Train[:,0], Peau_Train[:,1], '.r', label='Peau')
plt.legend()
plt.show()


# Les bases d’apprentissage et de test sont constituer une base de train de 1/1000 (reponse a verifier)
# Il y a deux classes
print("Train", np.shape(X_train))
print("Test", np.shape(X_test))
print("Peau (classe 1)", np.shape(Peau_Train))
print("Non_peau (classe 0)", np.shape(Nonpeau_Train))


# II. Modélisation de la vraisemblance des observations par une loi normale 2D avec des dimensions décorrélées
# a. Estimation de la vraisemblance des observations des pixels de teinte chaire

# Les variables sont indicées par Cb et Cr car on ne veut pas que la luminance Y impacte les données

mcb = np.mean(X_train[:, 0])
mcr = np.mean(X_train[:, 1])

scb = np.std(X_train[:, 0])
scr = np.std(X_train[:, 1])

p1 = norm(mcb, scb) 
p2 = norm(mcr, scr) 
p_train = p1.pdf(X_train[:,0]) * p2.pdf(X_train[:,1])

# mcb, mcr, scb et scb sont des scalaires donc de dimention 1
print("p_train", np.shape(p_train))


# La dimension de p pour un x donné est de 1
# L'hypothèse qui nous permet d’estimer la valeur de la loi normale
#  est que les dimension sont décorrélées

## b. Mise en place du classifieur

y_train_predit = np.zeros(len(p_train))
mp = np.mean(p_train)
for i in range(len(p_train)):
    if p_train[i] < mp: 
        y_train_predit[i] = 0
    else:
        y_train_predit[i] = 1

confusion_matrix(y_train, y_train_predit)

## c. Courbe ROC

SEUILS = np.linspace(np.min(p_train), np.max(p_train), 20) 

## d. Classification des pixels de test 

# III. Modélisation de la vraisemblance des observations par une loi normale 2D  

# IV. Test sur une nouvelle image