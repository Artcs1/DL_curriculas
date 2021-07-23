import numpy as np
import torch
from os import scandir, getcwd
from os.path import abspath,isfile


from torch.utils.data import Dataset


class Embedding(Dataset):

    def __init__(self, model='bert', sample='train'):
        data = np.load("./Embeddings/"+model+".npy")

        X   = data[:,:-1]
        Y   = data[:,-1]
        siz = X.shape[1]


        if sample != 'all_P':
            ind = np.where(Y!=5)
            X = X[ind]
            Y = Y[ind]

        num_train   = X.shape[0]
        split_valid = int(np.floor(0.6*num_train))
        split_test  = int(np.floor(0.8*num_train))

        num_types = np.zeros(num_train) + 4

        for i in range(5):
            c   = np.where(Y==i)
            ind = c[0]
            c   = c[0].shape[0]
            split_valid = int(np.floor(0.6*c))
            split_test  = int(np.floor(0.8*c))
            num_types[ind[:split_valid]] = 1
            num_types[ind[split_valid:split_test]] = 2
            num_types[ind[split_test:]] = 3

        if sample == 'train':
            new_X = np.empty((0,siz), np.float32)
            new_Y = np.array((), np.float32)
            for i in range(5):
                c = np.where(Y==i)
                ind = c[0]
                c   = c[0].shape[0]
                split_valid = int(np.floor(0.6*c))
                split_test  = int(np.floor(0.8*c))
                new_X = np.append(new_X, X[ind][:split_valid], axis=0)
                new_Y = np.append(new_Y, Y[ind][:split_valid], axis=0)
                num_types[ind]=1
            X = new_X
            Y = new_Y
        if sample == 'valid':
            new_X = np.empty((0,siz), np.float32)
            new_Y = np.array((), np.float32)
            for i in range(5):
                c = np.where(Y==i)
                ind = c[0]
                c   = c[0].shape[0]
                split_valid = int(np.floor(0.6*c))
                split_test  = int(np.floor(0.8*c))
                new_X = np.append(new_X, X[ind][split_valid:split_test], axis=0)
                new_Y = np.append(new_Y, Y[ind][split_valid:split_test], axis=0)
                num_types[ind]=2
            X = new_X
            Y = new_Y
        if sample == 'test':
            new_X = np.empty((0,siz), np.float32)
            new_Y = np.array((), np.float32)
            for i in range(5):
                c = np.where(Y==i)
                ind = c[0]
                c   = c[0].shape[0]
                split_valid = int(np.floor(0.6*c))
                split_test  = int(np.floor(0.8*c))
                new_X = np.append(new_X, X[ind][split_test:], axis=0)
                new_Y = np.append(new_Y, Y[ind][split_test:], axis=0)
                num_types[ind]=3
            X = new_X
            Y = new_Y


        types = []
        for i in range(num_train):
            if num_types[i] == 1:
                s = 'train'
            elif num_types[i] ==2:
                s = 'valid'
            elif num_types[i] ==3:
                s = 'test'
            else:
                s = 'peru'
            types.append(s)

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


