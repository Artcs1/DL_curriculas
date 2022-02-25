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

def main():

    parser = argparse.ArgumentParser(description = 'Curriculas')
    parser.add_argument("--model",default="bert_curso")
    parser.add_argument("--mode",default='AL')
    args = parser.parse_args()

    model = Net_A(768,128)
    checkpoint = torch.load('ml_bert_curso_128_64_7.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.cuda()
    model.eval()


    all_data  = Embedding(model="Embeddings/"+args.model, sample='all_P')
    print(all_data.Y)
    

    if torch.cuda.is_available():
        embedding = model(all_data.X.cuda())
        D = embedding.cpu().detach().numpy()
        gt   = all_data.Y.detach().numpy()
    
    else:
        embedding = model(all_data.X)
        D = embedding.detach().numpy()
        gt   = all_data.Y.detach().numpy()

    DATA = {'x':D, 'y':gt}
    with open("Embeddings/"+args.mode.lower()+"_"+args.model+'.npy','wb') as f:
         pickle.dump(DATA,f)
if __name__== '__main__':
    main()
