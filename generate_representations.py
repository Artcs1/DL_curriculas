from tqdm import tqdm
from os import scandir, getcwd
from os.path import abspath,isfile
import nltk
nltk.download('stopwords')
from sklearn import preprocessing
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import re
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import gensim.downloader as api
import matplotlib.patheffects as PathEffects
from gensim.models import KeyedVectors
from scipy.cluster.hierarchy import dendrogram, linkage
import seaborn as sns
from scipy.cluster.hierarchy import fcluster, leaves_list, set_link_color_palette
import argparse
import logging
from simpletransformers.language_representation import RepresentationModel
import torch
import pickle

from Loader.data_loader import Curriculas
from Loader.data_loader import toembedding
from Loader.data_loader import format_curriculas

def fashion_scatter(x, colors, names, model, paths):
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

    plt.savefig('results/Images/tse-'+model+'.png')


    names2 = paths
    c = np.random.randint(1,5,size=305)

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

    plt.legend(handles=[pop_a, pop_b, pop_c, pop_d, pop_e, pop_f])



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
    plt.savefig('results/Images/pca-'+model+'.png')



def text_prepare(text):
    """
        text: a string

        return: modified initial string
    """
    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
    STOPWORDS = set(stopwords.words('english'))

    text = text.lower()# lowercase text
    text = re.sub(REPLACE_BY_SPACE_RE," ",text)# replace REPLACE_BY_SPACE_RE symbols by space in text
    text = re.sub(BAD_SYMBOLS_RE,"",text)# delete symbols which are in BAD_SYMBOLS_RE from text
    text = re.sub('\s+'," ",text)
    text = " ".join([w for w in text.split(" ") if w not in STOPWORDS])# delete stopwords from text
    return text


def test_text_prepare(listCourses):
    l = []
    for i in listCourses:
        l2 = []
        for j in i:
            l2.append(text_prepare(j))
        l.append(l2)
    return l

def dendogram(X,gt,model):
    linked = linkage(X, 'ward')

    blanks = gt
    for i in range(len(blanks)):
        blanks[i]=''

    set_link_color_palette(["k", "y",'burlywood'])
    plt.figure(figsize=(5, 10))
    dendrogram(linked,leaf_rotation=0,orientation = 'left',leaf_font_size=8.,labels = blanks)
    ax = plt.gca()
    xlbls = ax.get_ymajorticklabels()

    plt.savefig('results/Images/dendogram-'+model+'.png')

def call_model(string):
    KeyedV = KeyedVectors.load(string)
    return KeyedV

def train_model(string):
    model = api.load(string)
    model.wv.save(string + ".kv")
    return model.wv

def main():

    parser = argparse.ArgumentParser(description = 'Curriculas')
    parser.add_argument("--model",default="word2vec100wv")
    parser.add_argument("--save" ,default=True)
    parser.add_argument("path")
    args = parser.parse_args()

    data_curriculas = Curriculas(path=args.path, data = "all")
    curriculas = format_curriculas(data_curriculas.x)
    paths_absolute = data_curriculas.get_names()
    paths_relative =[]
    for name in paths_absolute:
        paths_relative.append(name.split('/')[-1])


    print("     Reading mallas .....    ")


    curriculas = test_text_prepare(curriculas)

    #from sklearn.feature_extraction.text import CountVectorizer
    #vectorizer = CountVectorizer(min_df=0.05,max_df = 0.5)
    #vectorizer.fit_transform(curriculas)

    size = 0
    if args.model == 'bert':
        model = RepresentationModel(model_type="bert", model_name="bert-base-uncased", use_cuda = torch.cuda.is_available())
        size = 768
    elif args.model == 'cl_bert':
        model = RepresentationModel(model_type="bert", model_name= "Model/CL_bert/", use_cuda = torch.cuda.is_available())
        size = 768
    elif args.model == 'lm_bert':
        model = RepresentationModel(model_type="bert", model_name= "Model/LM_bert/", use_cuda = torch.cuda.is_available())
        size = 768
    elif args.model == 'ml_bert':
        model = RepresentationModel(model_type="bert", model_name= "Model/ML_bert/", use_cuda = torch.cuda.is_available())
        size = 768
    else:
        string = "./Model/" + args.model
        if isfile(string + ".kv"):
            model = call_model(string + ".kv")
        else:
            model = train_model(string)
        size = model.vectors.shape[1]

    print("     Generating embbedings .....     ")

    embedding = toembedding(curriculas, model, size, args.model)
    gt = np.asarray(data_curriculas.y)
    D = embedding

    ind_gt = np.where(gt!=5)[0]
    gt2 = gt[ind_gt]
    D2  = D[ind_gt]
    DATA = np.concatenate((D2,gt2.reshape(-1,1)), axis=1)
    if args.save == True:
        np.save("Embeddings/"+args.model,DATA)

    print("     Saving results .....     ")

    plotear(D.copy(),gt,['CS','CE','IT','IS','SE','Peru'],['red','green','blue','black','brown','purple'],args.model)

    fashion_tsne = TSNE(random_state=13).fit_transform(D.copy())
    fashion_scatter(fashion_tsne, gt,['CS','CE','IT','IS','SE','Peru'],args.model, paths_relative)

    dendogram(D, paths_relative,args.model)

    print("     Results saved   ")

if __name__ == "__main__":
    main()

