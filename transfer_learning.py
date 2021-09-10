from simpletransformers.language_modeling import LanguageModelingModel
import logging
import torch
import argparse
import numpy as np
import pandas as pd
from os import scandir, getcwd
from os.path import abspath,isfile
import argparse
import os

from Loader.data_loader import Curriculas
from Loader.data_loader import format_curriculas


def save_malla(all_data, path):
    f = open(path,'w')
    for curriculas in all_data:
        for c in curriculas:
            f.write(c+'\n')
    f.close()

def main():

    parser = argparse.ArgumentParser(description = 'Curriculas')
    parser.add_argument("--repetitions", default=10)
    parser.add_argument("path")
    args = parser.parse_args()

    if os.path.exists('./raw/train.txt'):
        traind = Curriculas(path=args.path, data = 'train')
        save_malla(format_curriculas(traind.x),'raw/train.txt')
        validd = Curriculas(path=args.path, data = 'valid')
        save_malla(format_curriculas(validd.x),'raw/eval.txt')
        testd  = Curriculas(path=args.path, data = 'test')
        save_malla(format_curriculas(testd.x) ,'raw/test.txt' )

    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)

    for ind in range(args.repetitions):
        train_args = {
            "save_eval_checkpoints": False,
            "num_train_epochs": 50,
            "save_model_every_epoch": False,
            "evaluate_during_training":True,
            "manual_seed": ind,
            "output_dir": "Model/LM_bert"+str(ind)+"/",
            "best_model_dir": "Model/LM_bert"+str(ind)+"/best_model",
        
        }
        
    
        model = LanguageModelingModel('bert', 'bert-base-cased', args=train_args, use_cuda = torch.cuda.is_available())
    
        model.train_model("raw/train.txt", eval_file="raw/eval.txt")
    
    #model.eval_model("raw/test.txt")

if __name__ == '__main__':
    main()
