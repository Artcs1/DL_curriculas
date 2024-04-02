import numpy as np
import torch
from os import scandir, getcwd
from os.path import abspath,isfile

from torch.utils.data import Dataset

from torch import nn, optim, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

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
        return x, alphas


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
        
        return X, alphas


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


