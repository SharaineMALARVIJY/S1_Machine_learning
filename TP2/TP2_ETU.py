import numpy as np
from matplotlib import pyplot as plt
from display import plotHistoClasses, plotGallery
import sklearn.model_selection as sk
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# from sklearn.metrics import confusion_matrix, accuracy_score

# #############################################################################
# I Chargement et mise en forme des données

print("I Chargement et mise en forme des données")
print("")

[X, y, name]=np.load('TP2.npy', allow_pickle=True)

# plotGallery(X)
# plotHistoClasses(y)

X_train, X_test, y_train, y_test = sk.train_test_split(X, y, random_state=543)


print("Nombre d’images X_train :", X_train.shape[0])
print("Nombre d’images X_test :", X_test.shape[0])
print("Nombre d’images y_train :", y_train.size)
print("Nombre d’images y_test :", y_test.size)
print("")
print("Dimensions X_train :", X_train.shape)
print("Dimensions X_test :", X_test.shape)
print("Dimensions y_train :", y_train.shape)
print("Dimensions y_test :", y_test.shape)

print("")
print("II Redimensionnement des données")
n=2914

X_train = X_train.reshape(X_train.shape[0], n)
X_test = X_test.reshape(X_test.shape[0], n)

print("Dimensions X_train redimensionner :", X_train.shape)
print("Dimensions X_test redimensionner :", X_test.shape)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = PCA(n_components=965)
model.fit(X_train)
x = np.linspace(0, 99, 100)
plt.plot(x, model.explained_variance_ratio_)
plt.show()