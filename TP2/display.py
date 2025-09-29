# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 08:31:51 2025

@author: MALARVIJY Sharaine 21206543
"""

import numpy as np
from matplotlib import pyplot as plt

def plotGallery(images, n=16, title=None):
    # Affiche les n premières images contenues dans images
    # images est de taille Nb image*Ny*Nx
    n = min(n, images.shape[0])
    nSubplots = int(np.ceil(np.sqrt(n)))
    fig, axs = plt.subplots(nSubplots, nSubplots)
    for i in range(n):
        axs[i // nSubplots, i % nSubplots].imshow(images[i], cmap=plt.cm.gray)
        axs[i // nSubplots, i % nSubplots].set_xticks([])
        axs[i // nSubplots, i % nSubplots].set_yticks([])
    if title:
        plt.suptitle(title)
    plt.show()

def plotHistoClasses(lbls):
    # Affiche le nombre d'exemples par classe
    nLbls = np.array([[i, np.where(lbls == i)[0].shape[0]] for i in np.unique(lbls)])
    plt.figure()
    plt.bar(nLbls[:, 0], nLbls[:, 1])
    plt.title("Nombre d'exemples par classe")
    plt.grid(axis='y')
    plt.show()
    