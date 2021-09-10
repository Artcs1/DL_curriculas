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
from pytorch_metric_learning.reducers import ThresholdReducer
from pytorch_metric_learning.regularizers import LpRegularizer
import hiddenlayer as hl

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
        self.X = tuple(d.X for d in self.datasets)
        self.Y = self.datasets[0].Y

    def __getitem__(self, index):
        return tuple(d.X[index] for d in self.datasets), self.datasets[0].Y[index]

    def __len__(self):
        return min(len(d) for d in self.datasets)

class Net_M(torch.nn.Module):

    def __init__(self, D_in, D_out):
        super(Net_M,self).__init__()
        self.linear1 = torch.nn.Linear(D_in, D_out)

    def forward(self, x):
        x = self.linear1(x) 
        return x
        #return f.normalize(x,dim=0,p=2)

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

class Net_F2(torch.nn.Module):

    def __init__(self, D_in, D_out):
        super(Net_F2, self).__init__()
        self.linear1 = torch.nn.Linear(100,1)
        self.linear2 = torch.nn.Linear(768,100)
        self.linear3 = torch.nn.Linear(128,100)
        self.softmax = torch.nn.Softmax(dim=1)
        
    def forward(self, x):
        glove_emb = x[0]
        word_emb  = x[1]
        bert_emb  = x[2]
        cbert_emb = x[3]
        lbert_emb = x[4]
        mbert_emb = x[5]
        abert_emb = x[6]
        bert_emb  = self.linear2(bert_emb)         
        cbert_emb = self.linear2(cbert_emb)
        lbert_emb = self.linear2(lbert_emb)
        mbert_emb = self.linear3(mbert_emb)
        abert_emb = self.linear3(abert_emb)

        scalar_glove = self.linear1(glove_emb)
        scalar_word  = self.linear1(word_emb)
        scalar_bert  = self.linear1(bert_emb)
        scalar_cbert = self.linear1(cbert_emb)
        scalar_lbert = self.linear1(lbert_emb)
        scalar_mbert = self.linear1(mbert_emb)
        scalar_abert = self.linear1(abert_emb)
        scalars      = torch.cat([scalar_glove, scalar_word, scalar_bert, scalar_cbert, scalar_lbert, scalar_mbert, scalar_abert],axis=1)

        X = torch.stack((glove_emb, word_emb, bert_emb, cbert_emb, lbert_emb, mbert_emb, abert_emb),axis = 2)
        X = torch.transpose(X,1,2)    
        alphas       = self.softmax(scalars) #(bs,3)
        
        X   = torch.mul(X,torch.unsqueeze(alphas,2)) #( numero curriculas, numero maximo de cursos, 768)
        X   = torch.sum(X,axis=1)


        
        
        return X



def train_model(epochs, dataloader, validloader, tam, dim_out, MO):

    best_lr   = 0
    best_loss = np.inf
    for lr in [0.1, 0.01, 0.001, 0.0001, 0.00001]:
        print(lr)
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

    print(best_lr)
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
    parser.add_argument("--model",default="bert")
    parser.add_argument("--mode",default='AF2')
    parser.add_argument("--epochs",default=2000)
    parser.add_argument("--batch",default=64, type=int)
    parser.add_argument("--dim_out",default=512, type=int)
    parser.add_argument("--repetitions", default = 10)
    args = parser.parse_args()


    os.system('mkdir -p Model/'+args.mode+'_bert')
    os.system('mkdir -p Embeddings')


    modes = {'ML':Net_M,'AF':Net_A,'AF':Net_F, 'AF2':Net_F2}


    for seed in range(args.repetitions):

        if args.mode  == 'N':

            train_data = Embedding(model=args.model, sample='train')
            valid_data = Embedding(model=args.model, sample='valid')
        
            train_loader = DataLoader(dataset=train_data, batch_size=args.batch, shuffle=True)
            valid_loader = DataLoader(dataset=valid_data, batch_size=args.batch, shuffle=True)

        elif args.mode == 'AF2':

            train_data_glove = Embedding(model='glove', sample = 'train')
            print(train_data_glove)
            valid_data_glove = Embedding(model='glove', sample = 'valid')

            train_data_word = Embedding(model='word2vec', sample = 'train')
            valid_data_word = Embedding(model='word2vec', sample = 'valid')

            train_data_bert = Embedding(model='bert', sample = 'train')
            valid_data_bert = Embedding(model='bert', sample = 'valid')

            train_data_clbert = Embedding(model='cl_bert', sample = 'train')
            valid_data_clbert = Embedding(model='cl_bert', sample = 'valid')

            train_data_lmbert = Embedding(model='lm_bert', sample = 'train')
            
            valid_data_lmbert = Embedding(model='lm_bert', sample = 'valid')

            train_data_mlbert = Embedding(model='ML_OP/ml_bert_128_64_'+str(seed), sample = 'train')
            valid_data_mlbert = Embedding(model='ML_OP/ml_bert_128_64_'+str(seed), sample = 'valid')

            train_data_albert = Embedding(model='AL/al_bert_curso_128_64_'+str(seed), sample = 'train')
            valid_data_albert = Embedding(model='AL/al_bert_curso_128_64_'+str(seed), sample = 'valid')

            train_loader = DataLoader(dataset=ConcatDataset(train_data_glove, train_data_word, train_data_bert, train_data_clbert, train_data_lmbert, train_data_mlbert, train_data_albert), batch_size=args.batch, shuffle=True)
            valid_loader = DataLoader(dataset=ConcatDataset(valid_data_glove, valid_data_word, valid_data_bert, valid_data_clbert, valid_data_lmbert, train_data_mlbert, train_data_albert) ,batch_size=args.batch, shuffle=True)
            
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
        model, optimizer = train_model(args.epochs, train_loader, valid_loader, tam, args.dim_out, M)

        torch.save({
                'epoch': args.epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, 'Model/'+args.mode+'_bert/ml_'+args.model+'_'+str(args.dim_out)+'_'+str(args.batch)+'_'+str(seed)+'.pth')
       
        if args.mode   == 'N': 
            all_data  = Embedding(model=args.model, sample='all_P')
        elif args.mode == 'AF2':
            data_glove = Embedding(model='glove', sample = 'all_P')
            data_word = Embedding(model='word2vec', sample = 'all_P')
            data_bert = Embedding(model='bert', sample = 'all_P')
            data_clbert = Embedding(model='cl_bert', sample = 'all_P')
            data_lmbert = Embedding(model='lm_bert', sample = 'all_P')
            data_mlbert = Embedding(model='ML_OP/ml_bert_128_64_'+str(seed), sample = 'all_P')
            data_albert = Embedding(model='AL/al_bert_curso_128_64_'+str(seed), sample = 'all_P')
      
            all_data = ConcatDataset(data_glove, data_word, data_bert, data_clbert, data_lmbert, data_mlbert, data_albert) 
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
 
