import torch
import copy
import argparse
import torch.nn.functional as f
import pickle

from pytorch_metric_learning import miners, losses

from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pdb
import os

from torch.utils.data import Dataset
from Loader.emb_loader import Embedding
from Loader.model_loader import Net_A, Net_F, TransformerModel

from Loader.data_loader import Curriculas
from Loader.data_loader import toembedding
from Loader.data_loader import toembedding2
from Loader.data_loader import format_curriculas


def main():

    parser = argparse.ArgumentParser(description = 'Curriculas')
    parser.add_argument("--model",default="bert_curso")
    parser.add_argument("--mode",default='AL')
    args = parser.parse_args()

    for i in [7]:
        model = Net_A(768,128) # match output dimensions of the trained model
        #checkpoint = torch.load('Model/AL_bert_1e5/ml_bert_curso_128_64_'+str(i)+'.pth') # choose a model
        checkpoint = torch.load('Model/AL_bert/ml_bert_curso_128_64_'+str(i)+'.pth') # choose a model

        checkpoint =         model.load_state_dict(checkpoint['model_state_dict'])
        model = model.cuda()
        model.eval()

        data_curriculas = Curriculas(path='DATA_TG100', data = "all")
        curriculas = format_curriculas(data_curriculas.x)
        paths_absolute = data_curriculas.get_names()
        paths_relative = []
        for name in paths_absolute:
            paths_relative.append(name.split('/')[-1])


        all_data  = Embedding(model=args.model, sample='all_P')
        

        if torch.cuda.is_available():
            embedding, alphas = model(all_data.X.cuda())
            D = embedding.cpu().detach().numpy()
            gt   = all_data.Y.detach().numpy()
        
        else:
            embedding, alphas = model(all_data.X)
            D = embedding.detach().numpy()
            gt   = all_data.Y.detach().numpy()

        for ind in range(290):
            ids = (-alphas[ind]).argsort()[:5]
            for idx in ids:
                print(curriculas[ind][idx])


        DATA = {'x':D, 'y':gt}
        with open("Embeddings/"+args.mode.lower()+"_"+args.model+str(i)+'.npy','wb') as f:
             pickle.dump(DATA,f)

if __name__== '__main__':
    main()
