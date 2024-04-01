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

from pytorch_metric_learning.distances import CosineSimilarity
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from pytorch_metric_learning.reducers import ThresholdReducer
from pytorch_metric_learning.regularizers import LpRegularizer
import math
#import hiddenlayer as hl

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
        self.X = tuple(d.X for d in self.datasets)
        self.Y = self.datasets[0].Y

    def __getitem__(self, index):
        return tuple(d.X[index] for d in self.datasets), self.datasets[0].Y[index]

    def __len__(self):
        return min(len(d) for d in self.datasets)

class Net_A(torch.nn.Module):

    def __init__(self, D_in, D_out):
        super(Net_A,self).__init__()
        self.linear1 = torch.nn.Linear(D_in, 1)
        self.linear2 = torch.nn.Linear(D_in, D_out)
        self.softmax = torch.nn.Softmax(dim=1)
    def forward(self, x):
        # x      -> (numero curriculas, numero maximo de cursos, embedding)
        mask_x = torch.sum(x,axis=2)
        # mask_x -> (numero curriculas, numero maximo de cursos) 
        mask_x = torch.where(mask_x==0,0,1) # (numero curriculas, numero maximos de cursos)
        x_c = self.linear1(x)  # (numero curriculas, numero maximo de cursos, 1)
        x_c = torch.squeeze(x_c) # (numero curriculas, numero maximo de cursos)
        x_c = torch.mul(x_c,mask_x) # (numero curriculas, numero maximo de cursos)

        exps        = torch.exp(x_c)
        masked_exps = exps * mask_x.float()
        masked_sums = masked_exps.sum(1, keepdim=True) + 1e-5
        alphas      = masked_exps/masked_sums #( numero curriculas, numero maximo de cursos)

        #s_c = self.softmax(x_c)

        A   = torch.unsqueeze(alphas,2) # (numero curriculas, numero maximo de cursos, 1)    
        x   = torch.mul(x,torch.unsqueeze(alphas,2)) #( numero curriculas, numero maximo de cursos, 768)
        x   = torch.sum(x,axis=1)



        x   = self.linear2(x) 
        return x

class Net_F(torch.nn.Module):

    def __init__(self, D_in, D_out):
        super(Net_F, self).__init__()
        self.linear1 = torch.nn.Linear(100,1)
        self.linear2 = torch.nn.Linear(768,100)
        self.softmax = torch.nn.Softmax(dim=1)
        
    def forward(self, x):
        glove_emb = x[0]
        word_emb  = x[1]
        bert_emb  = x[2]
        bert_emb  = self.linear2(bert_emb)         

        scalar_glove = self.linear1(glove_emb)
        scalar_word  = self.linear1(word_emb)
        scalar_bert  = self.linear1(bert_emb)
        scalars      = torch.cat([scalar_glove, scalar_word, scalar_bert],axis=1)

        X = torch.stack((glove_emb,word_emb,bert_emb),axis = 2)
        X = torch.transpose(X,1,2)    
        alphas       = self.softmax(scalars) #(bs,3)
        
        X   = torch.mul(X,torch.unsqueeze(alphas,2)) #( numero curriculas, numero maximo de cursos, 768)
        X   = torch.sum(X,axis=1)


        
        return X


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        print(x)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)



class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int, nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor):#, src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        print(src.shape)
        src = src.type('torch.LongTensor').cuda()
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        #output = self.transformer_encoder(src, src_mask)
        return src


def train_model(epochs, dataloader, validloader, tam, dim_out, MO, mode):

    print(MO)


    best_lr   = 0
    best_loss = np.inf
    for lr in [0.1, 0.01, 0.001, 0.0001, 0.00001]:
        if mode == 'T':
            model = MO(498,768,1,100,1)
        else:
            model = MO(tam, dim_out)
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
            best_loss = final_loss

    #best_lr = 1e-5

    if mode == 'T':
        model = MO(498,768,1,100,1)   
    else:
        model = MO(tam, dim_out)
      

    if torch.cuda.is_available():
        model = model.cuda()
    model.train()

    optimizer = optim.SGD(model.parameters(), lr=best_lr, momentum=0.9)
    loss_func = losses.TripletMarginLoss(distance = CosineSimilarity(),embedding_regularizer = LpRegularizer())


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

    #fig, ax = plt.subplots()
    #ax.plot(L,label = 'train')
    #ax.plot(V,label = 'valid')
    #ax.legend()
    #plt.show()
    return best_model, optimizer



def main():


    parser = argparse.ArgumentParser(description = 'Curriculas')
    parser.add_argument("--model",default="bert_curso")
    parser.add_argument("--mode",default='N')
    parser.add_argument("--epochs",default=1000)
    parser.add_argument("--batch",default=4, type=int)
    parser.add_argument("--dim_out",default=512, type=int)
    parser.add_argument("--repetitions", default = 10)
    args = parser.parse_args()


    os.system('mkdir -p Model/'+args.mode+'_bert')
    os.system('mkdir -p Embeddings')


    modes = {'N':Net_A,'AF':Net_F, 'T':TransformerModel}


    for seed in range(args.repetitions):

        if args.mode  == 'N' or args.mode == 'T':

            train_data = Embedding(model=args.model, sample='train')
            valid_data = Embedding(model=args.model, sample='valid')
        
            train_loader = DataLoader(dataset=train_data, batch_size=args.batch, shuffle=True)
            valid_loader = DataLoader(dataset=valid_data, batch_size=args.batch, shuffle=True)

            print(train_data.X.shape)

        elif args.mode == 'AF':
            train_data_glove = Embedding(model='glove', sample = 'train')
            valid_data_glove = Embedding(model='glove', sample = 'valid')

            train_data_word = Embedding(model='word2vec', sample = 'train')
            valid_data_word = Embedding(model='word2vec', sample = 'valid')


            train_data_bert = Embedding(model='bert', sample = 'train')
            valid_data_bert = Embedding(model='bert', sample = 'valid')

            train_loader = DataLoader(dataset=ConcatDataset(train_data_glove, train_data_word, train_data_bert), batch_size=args.batch, shuffle=True)
            valid_loader = DataLoader(dataset=ConcatDataset(valid_data_glove, valid_data_word, valid_data_bert) ,batch_size=args.batch, shuffle=True)


        if 'bert' in args.model:
            tam = 768
        else:
            tam = 100     
        

        torch.manual_seed(seed)
        M = modes[args.mode]
        model, optimizer = train_model(args.epochs, train_loader, valid_loader, tam, args.dim_out, M, args.mode)

        torch.save({
                'epoch': args.epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, 'Model/'+args.mode+'_bert/ml_'+args.model+'_'+str(args.dim_out)+'_'+str(args.batch)+'_'+str(seed)+'.pth')
       
        if args.mode   == 'N': 
            all_data  = Embedding(model=args.model, sample='all_P')
        
        elif args.mode == 'AF':
            data_glove = Embedding(model='glove', sample = 'all_P')
            data_word = Embedding(model='word2vec', sample = 'all_P')
            data_bert = Embedding(model='bert', sample = 'all_P')

            all_data = ConcatDataset(data_glove, data_word, data_bert)
        
        
        if torch.cuda.is_available():
            embedding = model(all_data.X.cuda())
            #DATA = np.concatenate((embedding.cpu().detach().numpy(), all_data.Y.detach().numpy().reshape(-1,1)), axis =1)
            D = embedding.cpu().detach().numpy()
            gt   = all_data.Y.detach().numpy()
        else:
            embedding = model(all_data.X)
            #DATA = np.concatenate((embedding.detach().numpy(),all_data.Y.detach().numpy().reshape(-1,1)), axis=1)
            D = embedding.detach().numpy()
            gt   = all_data.Y.detach().numpy()

        DATA = {'x':D, 'y':gt}
        with open("Embeddings/"+args.mode.lower()+"_"+args.model+'_'+str(args.dim_out)+'_'+str(args.batch)+'_'+str(seed)+'.npy','wb') as f:
            pickle.dump(DATA,f)



if __name__ == '__main__':
    main()
 
