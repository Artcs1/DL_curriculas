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
from Loader.data_loader import toembedding2
from Loader.data_loader import format_curriculas


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
    parser.add_argument("--model", default="glove")
    parser.add_argument("--mode" , default="curricula")
    parser.add_argument("--save" , default=True)
    parser.add_argument("path")
    args = parser.parse_args()

    data_curriculas = Curriculas(path=args.path, data = "all")
    curriculas = format_curriculas(data_curriculas.x)
    paths_absolute = data_curriculas.get_names()
    paths_relative = []
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
        model = RepresentationModel(model_type="bert", model_name= "Model/CL_bert4/best_model", use_cuda = torch.cuda.is_available())
        size = 768
    elif args.model == 'lm_bert':
        model = RepresentationModel(model_type="bert", model_name= "Model/LM_bert4/best_model", use_cuda = torch.cuda.is_available())
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
    
    if args.mode == 'curricula':
        embedding = toembedding(curriculas, model, size, args.model)
    else:
        embedding = toembedding2(curriculas, model, size, args.model)    
        args.model = args.model + '_curso'

    gt = np.asarray(data_curriculas.y)
    D  = embedding
    
    if args.save == True:
        DATA = {'x':D,'y':gt}
        with open('Embeddings/'+args.model+'.npy', 'wb') as f:
            pickle.dump(DATA, f)

    #ind_gt = np.where(gt!=5)[0]
    #gt2 = gt[ind_gt]
    #D2  = D[ind_gt]
    #DATA   = np.concatenate((D2,gt2.reshape(-1,1)), axis=1)
    #DATA_P = np.concatenate((D,gt.reshape(-1,1)), axis=1)
    #if args.save == True:
    #    #np.save("Embeddings/"+args.model,DATA)
    #    np.save("Embeddings/"+args.model+"_P",DATA_P)


    #print("     Saving results .....     ")

    #plotear(D.copy(),gt,['CS','CE','IT','IS','SE','Peru'],['red','green','blue','black','brown','purple'],args.model)

    #print(paths_relative)
    #fashion_tsne = TSNE(random_state=13).fit_transform(D.copy())
    #fashion_scatter(fashion_tsne, gt,['CS','CE','IT','IS','SE','Peru'],args.model, paths_relative)

    #dendogram(D, paths_relative,args.model)

    #print("     Results saved   ")

if __name__ == "__main__":
    main()

