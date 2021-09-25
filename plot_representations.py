from tqdm import tqdm
from os import scandir, getcwd
from sklearn import preprocessing
import pandas as pd
import numpy as np
import re
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import gensim.downloader as api
import matplotlib.patheffects as PathEffects
from scipy.cluster.hierarchy import dendrogram, linkage
import seaborn as sns
from scipy.cluster.hierarchy import fcluster, leaves_list, set_link_color_palette
import argparse
import logging
import torch
import pickle

from Loader.data_loader import Curriculas
from Loader.data_loader import toembedding
from Loader.data_loader import format_curriculas
from Loader.emb_loader import Embedding
from Loader.config import emb_size

from torch.utils.data import Dataset


def fashion_scatter(x, colors, names, model,paths):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))
    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot()
    for i,cx in zip(np.unique(colors),names):
        ids1 = np.where(colors==i)
        sc = ax.scatter(x[ids1,0], x[ids1,1], lw=0, s=40, c=palette[colors[ids1].astype(np.int)], label = cx)
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)

    # produce a legend with the unique colors from the scatter
    ax.legend(loc='upper left',frameon = True)
    #ax.axis('off')
    ax.axis('tight')

    plt.show()

    names2 = paths
    c = np.random.randint(1,5,size=len(names2))

    norm = plt.Normalize(1,20)
    cmap = plt.cm.RdYlGn

    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot()

    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)])


    pop_a =  mpatches.Patch(color=palette[0], label='CS')
    pop_b =  mpatches.Patch(color=palette[1], label='CE')
    pop_c =  mpatches.Patch(color=palette[2], label='IT')
    pop_d =  mpatches.Patch(color=palette[3], label='IS')
    pop_e =  mpatches.Patch(color=palette[4], label='SE')
    pop_f =  mpatches.Patch(color=palette[5], label='PERU')
    pop_g =  mpatches.Patch(color=palette[6], label='LATAM')

    plt.legend(handles=[pop_a, pop_b, pop_c, pop_d, pop_e, pop_f, pop_g])



    annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def update_annot(ind):

        pos = sc.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        text = "{}".format(" ".join([names2[n] for n in ind["ind"]]))
        annot.set_text(text)
        annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
        annot.get_bbox_patch().set_alpha(0.4)


    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = sc.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)

    plt.show()



def plotear(X,y,nombres,color,model):

    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)

    x_min, x_max = X_reduced[:, 0].min() - .2, X_reduced[:, 0].max() + .5
    y_min, y_max = X_reduced[:, 1].min() - .5, X_reduced[:, 1].max() + .5

    X_reduced[49:50,0] =x_min
    X_reduced[49:50,1] =y_min
    plt.figure(figsize=(8, 8))
    ax = plt.subplot()

    clases = np.unique(y)

    for i,j,c in zip(clases,nombres,color):
        ids1 = np.where(y==i)
        ax.plot(X_reduced[ids1[0], 0], X_reduced[ids1[0], 1], 'o', markersize=7, color=c, alpha=0.5, label=j)

    plt.xlim([x_min, x_min + 5])
    plt.ylim([y_min, y_min + 5])
    ax.legend(loc='upper left')

    ax.axis('tight')
    plt.plot()
    #plt.savefig('results/Images/pca-'+model+'.png')

def main():

    parser = argparse.ArgumentParser(description = 'Curriculas')
    parser.add_argument("--sample", default='all')
    parser.add_argument("model")
    args = parser.parse_args()

    #data_curriculas = Curriculas(path="DATA_TG100/", data = "all")
    #paths_absolute = data_curriculas.get_names()
    #paths_relative =[]
    #for name in paths_absolute:
    #    paths_relative.append(name.split('/')[-1])

    if args.sample !='all_P':
        classes = ['CS','CE','IT','IS','SE']
        color   = ['red','green','blue','black','brown']
    else:
        classes = ['CS','CE','IT','IS','SE','PE','LATAM']
        color   = ['red','green','blue','black','brown','purple','pink']

    data = Embedding(model=args.model, sample = args.sample)
    D  = data.X.numpy()
    print(D)
    gt = data.Y.numpy()


    print("     Saving results .....     ")

    plotear(D.copy(),gt,classes,color,args.model)

    fashion_tsne = TSNE(random_state=13).fit_transform(D.copy())

    print(data.types)

    fashion_scatter(fashion_tsne, gt,classes,args.model,data.types)

    #dendogram(D, paths_relative,args.model)

    print("     Results saved   ")

if __name__ == "__main__":
    main()


