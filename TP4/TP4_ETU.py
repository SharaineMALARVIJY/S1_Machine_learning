import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score



def visualize_classifier(model, X, y):
    ax = plt.gca()
    # Plot the training points
    ax.scatter(X[:, 0], X[:, 1], c=y, s=1, cmap='rainbow',
               clim=(y.min(), y.max()), zorder=3)
    ax.axis('tight')
    ax.axis('off')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(*xlim, num=200),
                         np.linspace(*ylim, num=200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    # Create a color plot with the results

    n_classes = len(np.unique(y))
    plt.scatter(xx.ravel(), yy.ravel(), c=Z, s=0.1, cmap='rainbow');
    ax.set(xlim=xlim, ylim=ylim)
    plt.show()
    
data = np.load("TP4.npz")
X_train, y_train, X_test, y_test = (data[key] for key in ["X_train", "y_train", "X_test", "y_test"])


plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=1, cmap='rainbow');
plt.show()  


#%% I. Chargement et visualisation des données

print(f"Il y a {X_train.shape[0]} points dans la base d’apprentissage")
print(f"Il y a {X_test.shape[0]} points dans la base d’apprentissage")

print(f"La dimension des données sont de {X_train.shape} pour la base d’apprentissage")
print(f"La dimension des données sont de {X_test.shape} pour la base d’test")


# II. Arbre de décision 
# a. Principe des arbres de décision

tree1 = tree.DecisionTreeClassifier(criterion='entropy', max_depth = 3, random_state= 543) 
tree1.fit(X_train, y_train) 
visualize_classifier(tree1, X_train, y_train) 
 
tree.plot_tree(tree1) 
plt.show() 
text_representation = tree.export_text(tree1) 
print(text_representation)

# b. Performance d’un classifieur multi-classes

y_pred = tree1.predict(X_test) 
C = confusion_matrix(y_test, y_pred) 
print(classification_report(y_test, y_pred)) 
print('Accuracy=', accuracy_score(y_test, y_pred)) 