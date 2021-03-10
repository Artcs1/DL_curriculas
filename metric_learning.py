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


def get_embedding(data, model, tokenizer, S):

    D = torch.zeros(S, 768)

    for ind, curricula in enumerate(data):
        curr_emb = torch.zeros(768)
        for cursos in curricula:
            text = cursos
            marked_text = "[CLS] " + text + " [SEP]"
            tokenized_text = tokenizer.tokenize(marked_text)
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
            tokens_tensor = torch.tensor([indexed_tokens])
            model.eval()

            outputs = model(tokens_tensor)
            word_emb = torch.mean(torch.squeeze(outputs[0]),0)
            curr_emb+=word_emb

        D[ind,:] = curr_emb/len(curricula)

    return D


def train_model(model_name, epochs, dataloader, lr):

    size = 0


    bert_config = BertConfig.from_pretrained('bert-base-uncased')
    tokenizer   = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case = False)
    if model_name == 'bert':
        model = BertModel.from_pretrained('bert-base-uncased', config = bert_config)
    elif model_name == 'cl_bert':
        model = BertModel.from_pretrained('Model/CL_bert/', config = bert_config)
    elif model_name == 'lm_bert':
        model = BertModel.from_pretrained('Model/LM_bert/', config = bert_config)
    elif model_name == 'ml_bert':
        model = BertModel.from_pretrained('Model/ML_bert/', config = bert_config)


    model2 = RepresentationModel(model_type="bert", model_name="bert-base-uncased", use_cuda = torch.cuda.is_available())
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
    L_loss=[]
    for epoch in range(epochs):
        sum_loss=0
        for i, (data, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            emb = format_curriculas(data)
            embeddings = get_embedding(emb, model, tokenizer, len(emb))
            loss = loss_func(embeddings, labels)
            #print(loss.item())
            sum_loss+=loss.item()
            loss.backward()
            optimizer.step()
        L_loss.append(sum_loss)
        #print(sum_loss)

    model.save_pretrained('./Model/ML_bert')
    bert_config.save_pretrained('./Model/ML_bert')
    tokenizer.save_pretrained('./Model/ML_bert')

def main():

    parser = argparse.ArgumentParser(description = 'Curriculas')
    parser.add_argument("--model",default="bert")
    parser.add_argument("--epochs",default=50)
    parser.add_argument("--batch",default=32)
    parser.add_argument("--lr",default=0.001)
    parser.add_argument("path")
    args = parser.parse_args()

    train_data = Curriculas(path=args.path, data = 'train')
    val_data   = Curriculas(path=args.path, data = 'valid')

    train_loader = DataLoader(dataset=train_data, batch_size=args.batch, shuffle=True)
    val_loader   = DataLoader(dataset=val_data, batch_size=args.batch, shuffle=True)

    model = train_model(args.model, args.epochs, train_loader, args.lr)
if __name__ == '__main__':
        main()
