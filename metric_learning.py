from simpletransformers.language_representation import RepresentationModel
from transformers import BertModel
from transformers import BertConfig
from transformers import BertTokenizer
import torch
import argparse

from pytorch_metric_learning import losses

from Loader.data_loader import Curriculas
from Loader.data_loader import format_curriculas
from Loader.data_loader import toembedding


from torch import nn, optim
from torch.utils.data import DataLoader
from simpletransformers.language_representation import RepresentationModel
import numpy as np



def train_model(model_name, epochs, dataloader, lr):

    size = 0


    bert_config = BertConfig.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case = False)
    if model_name == 'bert':
        model2 = RepresentationModel(model_type="bert", model_name="bert-base-uncased", use_cuda = torch.cuda.is_available())
        model = BertModel.from_pretrained('bert-base-uncased', config = bert_config)
        size = 768
    elif model_name == 'cl_bert':
        model2 = RepresentationModel(model_type="bert", model_name= "Model/CL_bert/", use_cuda = torch.cuda.is_available())
        model = BertModel.from_pretrained('Model/CL_bert/', config = bert_config)
        size = 768
    elif model_name == 'lm_bert':
        model2 = RepresentationModel(model_type="bert", model_name= "Model/LM_bert/", use_cuda = torch.cuda.is_available())
        model = BertModel.from_pretrained('Model/LM_bert/', config = bert_config)
        size = 768
    elif model_name == 'ml_bert':
        model2 = RepresentationModel(model_type="bert", model_name= "Model/ML_bert/", use_cuda = torch.cuda.is_available())
        model = BertModel.from_pretrained('Model/ML_bert/', config = bert_config)
        size = 768

    model.train()
    from transformers import AdamW
    optimizer = AdamW(model.parameters(), lr=lr)

    no_decay = ['bias', 'LayerNorm.weight']
    #optimizer_grouped_parameters = [
    #    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    #    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    #]
    #optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)

    #optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    loss_func = losses.TripletMarginLoss()
    for epoch in range(epochs):
        for i, (data, labels) in enumerate(dataloader):
            print(i)
            optimizer.zero_grad()
            emb = format_curriculas(data)
            embeddings = toembedding(emb, model2, size, model)
            embeddings = torch.from_numpy(embeddings)
            embeddings.requires_grad = True
            loss = loss_func(embeddings, labels)
            loss.backward()
            optimizer.step()

    model.save_pretrained('./Model/ML_bert')
    bert_config.save_pretrained('./Model/ML_bert')
    tokenizer.save_pretrained('./Model/ML_bert')

def main():

    parser = argparse.ArgumentParser(description = 'Curriculas')
    parser.add_argument("--model",default="bert")
    parser.add_argument("--epochs",default=50)
    parser.add_argument("--batch",default=2)
    parser.add_argument("--lr",default=0.00001)
    parser.add_argument("path")
    args = parser.parse_args()

    train_data = Curriculas(path=args.path, data = 'train')
    val_data   = Curriculas(path=args.path, data = 'valid')

    train_loader = DataLoader(dataset=train_data, batch_size=2)
    val_loader   = DataLoader(dataset=val_data, batch_size=2)

    model = train_model(args.model, args.epochs, train_loader, args.lr)
if __name__ == '__main__':
        main()
