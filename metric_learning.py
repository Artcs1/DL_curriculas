import torch
import copy
import argparse
import torch.nn.functional as f

from pytorch_metric_learning import miners, losses

from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pdb
import os

from torch.utils.data import Dataset
from Loader.emb_loader import Embedding

from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.reducers import ThresholdReducer
from pytorch_metric_learning.regularizers import LpRegularizer


class Net(torch.nn.Module):

    def __init__(self, D_in, D_out):
        super(Net,self).__init__()
        self.linear1 = torch.nn.Linear(D_in, D_out)

    def forward(self, x):
        x = self.linear1(x)
        return x
        #return f.normalize(x,dim=0,p=2)

def train_model(epochs, dataloader, validloader, tam, dim_out, model_name):

    best_lr   = 0
    best_loss = np.inf
    for lr in [0.1, 0.01, 0.001, 0.0001, 0.00001]:
        print(lr)
        model = Net(tam, dim_out)
        if torch.cuda.is_available():
            model = model.cuda()
        model.train()

        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        loss_func = losses.TripletMarginLoss(distance = CosineSimilarity(),
			 	     embedding_regularizer = LpRegularizer())
        for epoch in range(int(epochs/5)):
            for i, (X, Y) in enumerate(dataloader):
                optimizer.zero_grad()
                if torch.cuda.is_available():
                    X = X.cuda()
                    Y = Y.cuda()
                sub_X = model(X)
                loss  = loss_func(sub_X, Y)
                loss.backward()
                optimizer.step()
            valid_loss = 0
            for i, (X,Y) in enumerate(validloader):
                if torch.cuda.is_available():
                    X = X.cuda()
                    Y = Y.cuda()
                sub_X = model(X)
                loss  = loss_func(sub_X, Y)
                valid_loss += loss.item()
            final_loss = valid_loss/len(validloader)

        if best_loss > final_loss:
            best_lr = lr


    model = Net(tam, dim_out)
    if torch.cuda.is_available():
        model = model.cuda()
    model.train()

    optimizer = optim.SGD(model.parameters(), lr=best_lr, momentum=0.9)
    miner = miners.MultiSimilarityMiner()

    L = []
    V = []


    best_val = 1000000
    for epoch in range(epochs):
        cum_loss = 0
        for i, (X, Y) in enumerate(dataloader):
            optimizer.zero_grad()
            if torch.cuda.is_available():
                X = X.cuda()
                Y = Y.cuda()
            sub_X = model(X)
            hard_pairs = miner(sub_X, Y)
            loss = loss_func(sub_X, Y, hard_pairs)
            cum_loss += loss.item()
            loss.backward()
            optimizer.step()
        L.append(cum_loss/len(dataloader))
        valid_loss = 0
        for i, (X,Y) in enumerate(validloader):
            if torch.cuda.is_available():
                X = X.cuda()
                Y = Y.cuda()
            sub_X = model(X)
            hard_pairs = miner(sub_X, Y)
            loss = loss_func(sub_X, Y, hard_pairs)
            valid_loss += loss.item()
            if best_val > loss.item():
                best_val = loss.item()
                best_model = copy.deepcopy(model)
        V.append(valid_loss/len(validloader))

    fig, ax = plt.subplots()
    ax.plot(L,label = 'train')
    ax.plot(V,label = 'valid')
    ax.legend()
    plt.show()
    return best_model, optimizer

def main():


    parser = argparse.ArgumentParser(description = 'Curriculas')
    parser.add_argument("--model",default="bert_P")
    parser.add_argument("--epochs",default=1000)
    parser.add_argument("--batch",default=128)
    parser.add_argument("--dim_out",default=256)
    parser.add_argument("--repetitions", default = 1)
    args = parser.parse_args()


    os.system('mkdir -p Model/ML_bert')
    os.system('mkdir -p Embeddings')

    train_data = Embedding(model=args.model, sample='train')
    valid_data = Embedding(model=args.model, sample='valid')

    tam = train_data.X[0].shape[0]


    train_loader = DataLoader(dataset=train_data, batch_size=args.batch, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=args.batch, shuffle=True)

    for seed in range(args.repetitions):

        model, optimizer = train_model(args.epochs, train_loader, valid_loader, tam, args.dim_out, args.model)

        torch.save({
                'epoch': args.epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, 'Model/ML_bert/ml_'+args.model+'_'+str(args.dim_out)+'_'+str(args.batch)+'_'+str(seed)+'.pth')

        all_data  = Embedding(model=args.model, sample='all_P')
        embedding = model(all_data.X)

        DATA = np.concatenate((embedding.detach().numpy(),all_data.Y.detach().numpy().reshape(-1,1)), axis=1)
        np.save("Embeddings/ml_"+args.model+'_'+str(args.dim_out)+'_'+str(args.batch)+'_'+str(seed),DATA)



if __name__ == '__main__':
    main()
