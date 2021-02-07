from simpletransformers.language_modeling import LanguageModelingModel
import logging
import torch
import argparse
import numpy as np
import pandas as pd
from os import scandir, getcwd
from os.path import abspath,isfile
import argparse

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
    parser.add_argument("path")
    args = parser.parse_args()

    f = open('raw/train.txt','w')
    traind = Curriculas(path=args.path, data = 'train')
    save_malla(format_curriculas(traind.x),'raw/train.txt')
    validd = Curriculas(path=args.path, data = 'valid')
    save_malla(format_curriculas(validd.x),'raw/eval.txt')
    testd  = Curriculas(path=args.path, data = 'test')
    save_malla(format_curriculas(testd.x) ,'raw/test.txt' )

    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)

    train_args = {
        "reprocess_input_data": True,
        "overwrite_output_dir": True,
        "num_train_epochs": 2,

    }

    model = LanguageModelingModel('bert', 'bert-base-cased', args=train_args, use_cuda = torch.cuda.is_available())

    model.train_model("raw/train.txt", eval_file="raw/eval.txt")

    model.eval_model("raw/test.txt")

if __name__ == '__main__':
    main()
