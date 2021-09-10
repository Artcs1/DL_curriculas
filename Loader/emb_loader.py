import numpy as np
import torch
from os import scandir, getcwd
from os.path import abspath,isfile


from torch.utils.data import Dataset


class Embedding(Dataset):

    def __init__(self, model='bert', sample='train'):
        data = np.load("./Embeddings/"+model+".npy", allow_pickle=True)
        paths = open('./DATA_TG100/all.txt').read().splitlines()
        

        X   = data['x']
        Y   = data['y']
        siz = X.shape[1]
        
        if sample != 'all_P':
            ind = np.where(Y!=5)[0]
            X = X[ind]
            Y = Y[ind]
        
        length = []
        for i in range(5):
            length.append(np.where(Y==i)[0].shape[0])
        len_train = [ int(np.round(val * 0.6)) for val in length ]
        len_val   = [ int(np.round(val * 0.2)) for val in length ]
        len_test  = [ int(length[i] - len_train[i] - len_val[i]) for i in np.arange(len(length))]


        num_train   = X.shape[0]
        num_types = np.zeros(num_train) + 4

        for i in range(5):
            c   = np.where(Y==i)
            ind = c[0]
            num_types[ind[:len_train[i]]] = 1
            num_types[ind[len_train[i]:len_train[i]+len_val[i]]] = 2
            num_types[ind[len_train[i]+len_val[i]:]] = 3

        LX = []
        LY = []

        if sample == 'train':
            for i in range(5):
                c = np.where(Y==i)
                ind = c[0]
                LX.append(X[ind][:len_train[i]])
                LY.append(Y[ind][:len_train[i]])
                num_types[ind]=1
        if sample == 'valid':
            for i in range(5):
                c = np.where(Y==i)
                ind = c[0]
                LX.append(X[ind][len_train[i]:len_train[i]+len_val[i]])
                LY.append(Y[ind][len_train[i]:len_train[i]+len_val[i]])
                num_types[ind]=2
        if sample == 'test':
            for i in range(5):
                c = np.where(Y==i)
                ind = c[0]
                LX.append(X[ind][len_train[i]+len_val[i]:])
                LY.append(Y[ind][len_train[i]+len_val[i]:])
                num_types[ind]=3


        types = []
        for i in range(num_train):
            if num_types[i] == 1:
                s = 'train'
            elif num_types[i] ==2:
                s = 'valid'
            elif num_types[i] ==3:
                s = 'test'
            else:
                s = paths[i]
            types.append(s)

        #for ix in LX:
        #    print(ix.shape)
    
        #for iy in LY:
        #    print(iy.shape)

        if sample != 'all_P':
            X = np.concatenate(LX,axis=0)
            Y = np.concatenate(LY,axis=0)


        X = torch.from_numpy(X.astype(np.float32))
        Y = torch.from_numpy(Y.astype(np.float32))

        self.X    = X
        self.Y    = Y
        self.len  = X.shape[0]
        self.types = types

    def __getitem__(self, index):
        return self.X[index],self.Y[index]

    def __len__(self):
        return self.len


