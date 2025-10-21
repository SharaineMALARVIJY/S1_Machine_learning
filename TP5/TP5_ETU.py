# Sharaine MALARVIJY 
# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import mode


data = np.load("TP5.npz")
X_train, y_train, X_test, y_test = (data[key] for key in ["X_train", "y_train", "X_test", "y_test"])
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=1, cmap='rainbow');
plt.show()
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=1, cmap='rainbow');
plt.show()


print(f"Il y a {X_train.shape[0]} points dans la base d’apprentissage")
print(f"Il y a {X_test.shape[0]} points dans la base d’apprentissage")

print(f"La dimension des données sont de {X_train.shape} pour la base d’apprentissage")
print(f"La dimension des données sont de {X_test.shape} pour la base d’test")

nb_classe = round(max(y_test)+1)
print(f"Il y a {nb_classe} classe") 

for i in range(nb_classe):
    print(f"Effectif de {X_test[y_test==i].size} dans la classe {i}")

bias_list = []
var_list = []
prediction_list = []
Nrun = 30
for i in range(Nrun):
    X_train_run, X_test_run, y_train_run, y_test_run = train_test_split(X_train, y_train, train_size=0.6, random_state=42+i)
    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(X_train_run, y_train_run)
    prediction = neigh.predict(X_test_run)
    prediction_list.append(prediction)

#mode return la classe la plus predite pour les 30 run 
#count return le nbre de fois où la classe est prédite 

modes = mode(np.array(prediction_list), axis=0)
plus_pred = (modes.mode == modes.count)
 
print(bias_list)
print(var_list)
