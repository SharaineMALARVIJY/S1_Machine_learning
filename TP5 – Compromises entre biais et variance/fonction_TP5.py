import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import mode
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


def KPPV(X_train, y_train, X_test, k, Nrun): 
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
            bias_list.append(0)
        else :
            bias_list.append(1)
    
    bias = np.mean(bias_list)

    for c in modes.count:
        var_list.append(1-c/Nrun)
    variance = np.mean(var_list)
    return bias, variance

def affiche_bias_variance(axis_x, bias_list, variance_list, x_label):
    plt.figure()
    plt.plot(axis_x, bias_list, 'o-', color='blue')
    plt.title("Biais")
    plt.xlabel(x_label)
    plt.ylabel("Biais")
    plt.grid(True)


    plt.figure()
    plt.plot(axis_x, variance_list, 'o-', color='red')
    plt.title("Variance")
    plt.xlabel(x_label)
    plt.ylabel("Variance")
    plt.grid(True)


def DecisionTree(X_train, y_train, X_test, k, Nrun): 
    prediction_list = []
    for i in range(Nrun):
        X_train_run, _, y_train_run, _ = train_test_split(X_train, y_train, train_size=0.6, random_state=543+i)
        tree1 = tree.DecisionTreeClassifier(criterion='entropy', max_depth = k, random_state= 543) 
        tree1.fit(X_train_run, y_train_run)
        prediction = tree1.predict(X_test)
        prediction_list.append(prediction)   
    return prediction_list


def RandomForest(X_train, y_train, X_test, k, Nrun):
    prediction_list = []
    for i in range(Nrun):
        X_train_run, _, y_train_run, _ = train_test_split(X_train, y_train, train_size=0.6, random_state=543+i)
        RF = RandomForestClassifier(criterion='entropy', n_estimators=k, random_state=543, n_jobs=-1) 
        RF.fit(X_train_run, y_train_run)
        prediction = RF.predict(X_test)
        prediction_list.append(prediction)   
    return prediction_list
