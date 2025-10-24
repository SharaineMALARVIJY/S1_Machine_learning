# Sharaine MALARVIJY 
# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import mode
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier


data = np.load("TP5.npz")
X_train, y_train, X_test, y_test = (data[key] for key in ["X_train", "y_train", "X_test", "y_test"])
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=1, cmap='rainbow');
plt.show()
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=1, cmap='rainbow');
plt.show()

print("Dans la base de donnée TP5.npz")
print(f"Il y a {X_train.shape[0]} points dans la base d’apprentissage")
print(f"Il y a {X_test.shape[0]} points dans la base de test")

print(f"La dimension des données sont de {X_train.shape} pour la base d’apprentissage")
print(f"La dimension des données sont de {X_test.shape} pour la base de test")

nb_classe = round(max(y_test)+1)
print(f"Il y a {nb_classe} classe") 

for i in range(nb_classe):
    print(f"Effectif de {X_test[y_test==i].size} dans la classe {i}")

# -----------


def KPPV(X_train, y_train, k, Nrun): 
    prediction_list = []
    for i in range(Nrun):
        X_train_run, _, y_train_run, _ = train_test_split(X_train, y_train, train_size=0.6, random_state=543+i)
        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(X_train_run, y_train_run)
        prediction = neigh.predict(X_test)
        prediction_list.append(prediction)
    return prediction_list

#modes.mode return la classe la plus predite pour les 30 run 
#modes.count return le nbre de fois où la classe est prédite 

def calcul_bias_variance(prediction_list, y_test, Nrun) : 
    bias_list = []
    var_list = []
    modes = mode(np.array(prediction_list), axis=0)

    for i in range(len(y_test)):
        if modes.mode[i] == y_test[i]: 
            bias_list.append(1)
        else :
            bias_list.append(0)
    
    bias = np.mean(bias_list)

    for c in modes.count:
        var_list.append(1-c/Nrun)
    variance = np.mean(var_list)
    return bias, variance

Nrun = 30

prediction_list = KPPV(X_train, y_train, 1, Nrun)

bias, variance = calcul_bias_variance(prediction_list, y_test, Nrun) 

print("Pour k = 1 : ")
print(f"Biais = {bias}")
print(f"Variance = {variance}")

# III.

bias_list = []
variance_list = []

for i in range(1, 40):
    prediction_list = KPPV(X_train, y_train, i, Nrun)
    bias, variance = calcul_bias_variance(prediction_list, y_test, Nrun) 
    bias_list.append(bias)
    variance_list.append(variance)


def affiche_bias_variance(bias_list, variance_list, x_label):
    plt.figure()
    plt.plot(bias_list, 'o-', color='blue')
    plt.title("Biais")
    plt.xlabel(x_label)
    plt.ylabel("Biais")
    plt.grid(True)


    plt.figure()
    plt.plot(variance_list, 'o-', color='red')
    plt.title("Variance")
    plt.xlabel(x_label)
    plt.ylabel("Variance")
    plt.grid(True)

affiche_bias_variance(bias_list, variance_list, "K plus proche voisin")

# IV.


def DecisionTree(X_train, y_train, k, Nrun): 
    prediction_list = []
    for i in range(1, Nrun):
        X_train_run, _, y_train_run, _ = train_test_split(X_train, y_train, train_size=0.6, random_state=543+i)
        tree1 = tree.DecisionTreeClassifier(criterion='entropy', max_depth = k, random_state= 543) 
        tree1.fit(X_train_run, y_train_run)
        prediction = tree1.predict(X_test)
        prediction_list.append(prediction)   
    return prediction_list


bias_list = []
variance_list = []

for i in range(1, 40):
    prediction_list = DecisionTree(X_train, y_train, i, Nrun)
    bias, variance = calcul_bias_variance(prediction_list, y_test, Nrun) 
    bias_list.append(bias)
    variance_list.append(variance)

affiche_bias_variance(bias_list, variance_list, "Profondeur maximal de l'arbre de décision")


# %% V.

def RandomForest(X_train, y_train, k, Nrun):
    prediction_list = []
    for i in range(1, Nrun):
        X_train_run, _, y_train_run, _ = train_test_split(X_train, y_train, train_size=0.6, random_state=543+i)
        RF = RandomForestClassifier(criterion='entropy', n_estimators=k, random_state=543) 
        RF.fit(X_train_run, y_train_run)
        prediction = RF.predict(X_test)
        prediction_list.append(prediction)   
    return prediction_list


bias_list = []
variance_list = []
nb_arbre = list(range(1, 202, 20))

for i in nb_arbre:
    prediction_list = RandomForest(X_train, y_train, i, Nrun)
    bias, variance = calcul_bias_variance(prediction_list, y_test, Nrun) 
    bias_list.append(bias)
    variance_list.append(variance)

def affiche_bias_variance_RF(nb_arbre, bias_list, variance_list, title= "Nombre d'arbre du random forest"):
    plt.figure()
    plt.plot(nb_arbre,bias_list, 'o-', color='blue')
    plt.title("Biais")
    plt.xlabel(title)
    plt.ylabel("Biais")
    plt.grid(True)


    plt.figure()
    plt.plot(nb_arbre, variance_list, 'o-', color='red')
    plt.title("Variance")
    plt.xlabel(title)
    plt.ylabel("Variance")
    plt.grid(True)

affiche_bias_variance_RF(nb_arbre, bias_list, variance_list, title= "Nombre d'arbre du random forest")

# %% VI.



data = np.load("TP5a.npz")
X_train, y_train, X_test, y_test = (data[key] for key in ["X_train", "y_train", "X_test", "y_test"])
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=1, cmap='rainbow');
plt.show()
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=1, cmap='rainbow');
plt.show()

print("Dans la base de donnée TP5a.npz")
print(f"Il y a {X_train.shape[0]} points dans la base d’apprentissage")
print(f"Il y a {X_test.shape[0]} points dans la base de test")

print(f"La dimension des données sont de {X_train.shape} pour la base d’apprentissage")
print(f"La dimension des données sont de {X_test.shape} pour la base de test")

nb_classe = round(max(y_test)+1)
print(f"Il y a {nb_classe} classe") 

for i in range(nb_classe):
    print(f"Effectif de {X_test[y_test==i].size} dans la classe {i}")

